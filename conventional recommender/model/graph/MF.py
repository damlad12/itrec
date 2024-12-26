import torch
import torch.nn as nn
import pickle
import csv
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss, l2_reg_loss

class MF(GraphRecommender):
    def __init__(self, conf, training_set, test_set, valid_set, **kwargs):
        super(MF, self).__init__(conf, training_set, test_set, valid_set, **kwargs)
        self.model = Matrix_Factorization(self.data, self.emb_size)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb = rec_user_emb[user_idx]
                pos_item_emb = rec_item_emb[pos_idx]
                neg_item_emb = rec_item_emb[neg_idx]

                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) \
                             + l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) / self.batch_size

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 100 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)

        # After all epochs, store the best user/item embeddings
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

        # ------------------------------------------------------
        # 1) Create rating matrix
        # ------------------------------------------------------
        mf_pred = torch.matmul(self.user_emb, self.item_emb.transpose(0, 1))
        with open(f"{self.dataset_path}/rating_matrix.pkl", "wb") as f:
            pickle.dump(mf_pred.detach().cpu().numpy(), f)

        # ------------------------------------------------------
        # 2) Immediately generate mov_user.pkl / mov_item.pkl
        #    at the same time we create rating_matrix.pkl
        # ------------------------------------------------------
        user_map = self.data.user  # e.g. {"1": 0, "2": 1, ...}
        item_map = self.data.item  # e.g. {"10": 0, "50": 1, ...}

        with open(f"{self.dataset_path}/mov_user.pkl", "wb") as f:
            pickle.dump(user_map, f)
        with open(f"{self.dataset_path}/mov_item.pkl", "wb") as f:
            pickle.dump(item_map, f)

        # ------------------------------------------------------
        # 3) Save train_set_prediction.csv
        # ------------------------------------------------------
        self.save_predictions()

        # ------------------------------------------------------
        # 4) Save user_id_mapping.pkl, item_id_mapping.pkl, etc.
        # ------------------------------------------------------
        self.save_mappings()

        # ------------------------------------------------------
        # 5) If you also want mov_user_emb.pkl, mov_pred.pkl, etc.
        #    you can do it here as well, or inline with rating_matrix:
        # ------------------------------------------------------
        cm_user_emb = self.user_emb.detach().cpu().numpy()  # shape: (num_users, emb_size)
        cm_pred = mf_pred.detach().cpu().numpy()            # shape: (num_users, num_items)
        
        with open(f"{self.dataset_path}/mov_user_emb.pkl", "wb") as f:
            pickle.dump(cm_user_emb, f)
        with open(f"{self.dataset_path}/mov_pred.pkl", "wb") as f:
            pickle.dump(cm_pred, f)

        print("rating_matrix.pkl, mov_user.pkl, mov_item.pkl, train_set_prediction.csv, user_id_mapping.pkl, etc. all saved!")

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()

    def save_predictions(self):
        """
        Writes out 'train_set_prediction.csv' as [u, i, r, t].
        """
        user_map = self.data.user
        item_map = self.data.item
        id2user = {v: k for k, v in user_map.items()}
        id2item = {v: k for k, v in item_map.items()}

        with torch.no_grad():
            predictions = torch.matmul(self.user_emb, self.item_emb.transpose(0, 1)).cpu().numpy()

        csv_path = f"{self.dataset_path}/train_set_prediction.csv"
        with open(csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["u", "i", "r", "t"])
            for user_idx, user_predictions in enumerate(predictions):
                for item_idx, score in enumerate(user_predictions):
                    original_user_id = id2user[user_idx]
                    original_item_id = id2item[item_idx]
                    writer.writerow([original_user_id, original_item_id, score, 0])

    def save_mappings(self):
        """
        Writes out user_id_mapping.pkl, item_id_mapping.pkl, id2user_mapping.pkl, id2item_mapping.pkl.
        """
        user_map = self.data.user
        item_map = self.data.item

        with open(f"{self.dataset_path}/user_id_mapping.pkl", "wb") as f:
            pickle.dump(user_map, f)
        with open(f"{self.dataset_path}/item_id_mapping.pkl", "wb") as f:
            pickle.dump(item_map, f)
        with open(f"{self.dataset_path}/id2user_mapping.pkl", "wb") as f:
            pickle.dump(self.data.id2user, f)
        with open(f"{self.dataset_path}/id2item_mapping.pkl", "wb") as f:
            pickle.dump(self.data.id2item, f)


class Matrix_Factorization(nn.Module):
    def __init__(self, data, emb_size):
        super(Matrix_Factorization, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']
