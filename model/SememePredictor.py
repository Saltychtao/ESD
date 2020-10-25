from fairseq.data import Dictionary
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
import os
from fairseq import tokenizer, utils
from tqdm import tqdm
import numpy as np
import re
import pdb
import sys

sys.path.append(
    "/home/user_data55/lijh/pytorch-projects/VariationalDefinitionGeneration"
)
from sklearn.metrics import average_precision_score
from utils.metric import FScore, Accuracy


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


def build_word_dict(word_embed_path):
    word_dict = Dictionary()
    with open(word_embed_path,'r') as f:
        for line in f:
            word = line.split(' ',1)[0]
            word_dict.add_symbol(word)
    word_dict.finalize(padding_factor=1)
    return word_dict


def build_sememe_dict(datapath):
    sememe_dict = Dictionary()
    with open(os.path.join(datapath,'HowNet.edge'),'r') as f:
        for line in f:
            sememes = line.strip().split('\t')[1]
            for s in sememes.split():
                sememe_dict.add_symbol(s)
    sememe_dict.finalize(threshold=5,padding_factor=1)
    return sememe_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def precision_recall_fscore(pred, target):
    num_pred = 0.0
    num_correct = 0.0
    num_total = 0.0
    for p, t in zip(pred, target):
        num_pred += len(p)
        num_total += len(t)
        for s in t:
            if s in p:
                num_correct += 1
    return FScore(num_pred, num_correct, num_total)


def pad(data, pad_idx):
    max_len = max([len(instance) for instance in data])
    return [
        instance + [pad_idx] * max((max_len - len(instance), 0)) for instance in data
    ]


def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
    num_embedding = len(dictionary)
    padding_idx = dictionary.pad()
    embed_tokens = Embedding(num_embedding, embed_dim, padding_idx)
    embed_dict = utils.parse_embedding(embed_path)
    utils.print_embed_overlap(embed_dict, dictionary)
    # embed_keys = set(embed_dict.keys())
    # vocab_keys = set(dictionary.symbols))
    # print(vocab_keys - embed_keys)
    return utils.load_embedding(embed_dict, dictionary, embed_tokens), embed_dict


class SememePredictorDataset(Dataset):
    def __init__(self, word_dict, sememe_dict, split, args):
        super(SememePredictorDataset, self).__init__()
        self.data_path = args.data_path
        self.word_dict = word_dict
        self.sememe_dict = sememe_dict
        self.from_file(split)

    def from_file(self, split):
        self.instances = set()

        # with open(os.path.join(self.data_path, split + ".word"), "r") as fword, open(
        #     os.path.join(self.data_path, split + ".sememe"), "r"
        # ) as fsememe:
        #     for word, sememes in zip(fword, fsememe):
        #         word = word.rstrip()
        #         sememes = sememes.rstrip().split()
        #         self.instances.add(
        #             (
        #                 self.word_dict.index(word),
        #                 tuple([self.sememe_dict.index(s) for s in sememes]),
        #             )
        #         )
        with open(os.path.join(self.data_path,split+".txt")) as f:
            for line in f:
                word,sememes = tuple(line.strip().split('\t'))
                sememes = sememes.split()
                if word not in self.word_dict:
                    continue
                if len(sememes) == 1 and 'FuncWord' in sememes[0]:
                    continue
                self.instances.add(
                    (self.word_dict.index(word),tuple([self.sememe_dict.index(s) for s in sememes]))
                )

        self.examples = []
        for w, sememes in self.instances:
            self.examples.append((w, list(sememes)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "word": self.examples[idx][0],
            "sememes": self.examples[idx][1],
            "pad_idx": self.sememe_dict.pad(),
        }

    @staticmethod
    def collate_fn(list_of_instances):
        word = torch.tensor([x["word"] for x in list_of_instances])
        sememes = torch.tensor(
            pad(
                [x["sememes"] for x in list_of_instances],
                list_of_instances[0]["pad_idx"],
            )
        )
        return {"word": word, "sememes": sememes}


class SememeEncoder(nn.Module):
    def __init__(self, sememe_dict, word_dict, args):
        super(SememeEncoder, self).__init__()
        self.word_embed = args.pretrained_word_embedding
        self.set_vocab(sememe_dict, word_dict, 5)

    def set_vocab(self, sememe_vocab, word_vocab, max_word_len):
        sememe_to_word = torch.LongTensor(len(sememe_vocab), max_word_len)

        for i in range(len(sememe_vocab)):
            if i < sememe_vocab.nspecial:
                word_idxs = [word_vocab.pad()] * max_word_len
            else:
                sememe = sememe_vocab[i]
                english_part = sememe.split("|")[0]
                words = re.sub(r"([A-Z])", r" \1", english_part).split()
                words = [word_vocab.index(w.lower()) for w in words]
                word_idxs = [w for w in words] + [word_vocab.pad()] * (
                    max_word_len - len(words)
                )
            sememe_to_word[i] = torch.LongTensor(word_idxs)
        self.sememe_vocab = sememe_vocab
        self.sememe_to_word = sememe_to_word.to(device)

    def forward(self, sememes):
        # sememes: bsize * num_of_sememes
        words = self.sememe_to_word[sememes]
        words_mask = words != self.sememe_vocab.pad()
        word_embs = self.word_embed(
            words
        )  # bsize * num_of_sememes * max_words_per_sememe * D
        word_embs = word_embs * words_mask.float().unsqueeze(
            -1
        )  #  bsize * num_of_sememes * max_words_per_sememe * D
        word_embs = word_embs.sum(-2)  #  bsize * num_of_sememes * D
        return word_embs / (words_mask.float().sum(2).unsqueeze(-1) + 1e-6)

    def get_all_sememe_repre(self):
        with torch.no_grad():
            return self(
                torch.arange(len(self.sememe_vocab))
                .unsqueeze(0)
                .to(device)
            )


class SememePredictModel(nn.Module):
    def __init__(self, args):
        super(SememePredictModel, self).__init__()
        self.latent_M = args.latent_M
        self.latent_K = args.latent_K
        self.word_embed = args.pretrained_word_embedding
        self.pad_idx = args.sememe_dict.pad()
        # hidden_dim = 2*args.latent_M
        # hidden_dim = args.embed_dim
        # self.p_h = nn.Sequential(
        #     nn.Dropout(),
        #     Linear(args.embed_dim,hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     Linear(hidden_dim,args.embed_dim),
        #     nn.BatchNorm1d(args.embed_dim),
        #     nn.ReLU(),
        #     nn.Dropout()
        # )
        self.fc = Linear(args.embed_dim, args.latent_M,bias=False)
        # with torch.no_grad():
            # self.fc.weight.data[self.pad_idx].fill_(0)
        # self.fc = nn.Parameter(torch.FloatTensor(args.latent_M,args.embed_dim))
        # self.fc.weight.data = args.sememe_embedding
        # self.fc.weight.data.require_grad = False
        print(self)

    def forward(self, word):
        x = self.word_embed(word).detach()  # bsize * D
        if hasattr(self,'p_h'):
            hidden_repre = self.p_h(x)
        else:
            hidden_repre = x
        if hasattr(self, "sememe_embed"):
            sememe_repre = self.sememe_embed(torch.arange(self.latent_M).to(device))
            sememe_repre = self.upscale_fc(sememe_repre)  # NUM_SEMEME * hidden
            return torch.nn.functional.linear(hidden_repre, sememe_repre).view(
                -1, self.latent_M
            )
        else:
            return F.linear(hidden_repre,self.fc.weight).view(-1, self.latent_M)
            # return self.fc(hidden_repre).view(-1,self.latent_M)

    def compute_loss(self, word, sememes):
        logits = self(word)
        sememe_target = torch.zeros_like(logits).scatter_(-1, sememes, 1).float()
        sememe_target[:, self.pad_idx] = 0
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.reshape(-1), sememe_target.reshape(-1), reduction="none"
        ).view(sememe_target.size())
        # instance_mask = sememe_target[:,4:].sum(-1) > 0
        # instance_mask = (valid_sememe_target.sum(-1) == 0) == 0
        # num_valid_instances = instance_mask.sum(-1).float()
        # loss = (loss.mean(-1) * instance_mask.float()).sum(-1) / instance_mask.sum(-1)
        loss = loss.mean(-1).mean(-1)
        return loss

    def compute_metric(self, word, sememes, k):
        bsize = sememes.size()[0]
        scores = self.forward(word)
        sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
        correct = 0.0
        total = 0.0
        for index, sememe in zip(sorted_indices, sememes):
            total += (sememe != self.pad_idx).long().sum()
            index = set(index.tolist()[:k])
            sememe = set(sememe.tolist())
            correct += len(index.intersection(sememe))
        # return Accuracy(k * bsize, correct)
        return Accuracy(total.item(),correct)


def main(args):

    ########### Build Dictionary  ###############
    sememe_dict = build_sememe_dict(args.data_path)
    print("Total Sememes : {}".format(len(sememe_dict)))
    args.latent_M = len(sememe_dict)
    args.sememe_dict = sememe_dict

    word_dict = build_word_dict(args.embed_path)
    print("Total Words : {}".format(len(word_dict)))
    ############## Build Dataset ####################
    trainset = SememePredictorDataset(word_dict, sememe_dict, "train", args)
    validset = SememePredictorDataset(word_dict, sememe_dict, "valid", args)
    testset = SememePredictorDataset(word_dict, sememe_dict, "test", args)

    print("| Train: {}".format(len(trainset)))
    print("| Valid: {}".format(len(validset)))
    print("| Test: {}".format(len(testset)))

    train_iter = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        collate_fn=SememePredictorDataset.collate_fn,
    )
    valid_iter = DataLoader(
        validset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        collate_fn=SememePredictorDataset.collate_fn,
    )

    ############### Load Pretrained Word Embedding ##########
    # args.sememe_embedding = load_pretrained_embedding_from_file(args.sememe_embed_path,sememe_dict,args.embed_dim).to(device)
    args.pretrained_word_embedding, embed_dict = load_pretrained_embedding_from_file(
        args.embed_path, word_dict, args.embed_dim
    )
    args.pretrained_word_embedding.require_grad = False
    # args.sememe_embedding.require_grad = False
    sememe_encoder = SememeEncoder(sememe_dict, word_dict, args).to(device)
    sememe_embedding = sememe_encoder.get_all_sememe_repre().squeeze().detach()
    args.sememe_embedding = sememe_embedding
    ################# Build Model ##############
    model = SememePredictModel(args).to(device)
    # model = SememeFactorizationModel(args,sememe_dict.count).to(device)
    print("Model Built!")
    if args.mode == "analysis":
        model.load_state_dict(torch.load(args.save_path))
        analyze_sememe(
            valid_iter,
            model,
            word_dict,
            sememe_dict,
            embed_dict,
            "sememe_nearest_neighbour.txt",
        )
        analyze_word(
            valid_iter,
            model,
            word_dict,
            sememe_dict,
            embed_dict,
            "word_nearest_neighbour.txt",
        )
        exit()
    ############ Training #################
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    patience = 0
    best_metric = Accuracy()
    ################ Start Training ###########
    for epoch in range(1, args.epoch + 1):
        recall_at_8 = Accuracy()
        recall_at_16 = Accuracy()
        recall_at_32 = Accuracy()
        recall_at_64 = Accuracy()
        recall_at_128 = Accuracy()
        pbar = tqdm(train_iter, dynamic_ncols=True)
        pbar.set_description("[Epoch {}, Best Metric {}]".format(epoch, best_metric))
        model.train()
        for batch in pbar:
            word = batch["word"].to(device)
            sememes = batch["sememes"].to(device)
            # logits = model(word)
            loss = model.compute_loss(word, sememes)
            recall_at_8 += model.compute_metric(word, sememes, k=8)
            recall_at_16 += model.compute_metric(word, sememes, k=16)
            recall_at_32 += model.compute_metric(word, sememes, k=32)
            recall_at_64 += model.compute_metric(word,sememes,k=64)
            recall_at_128 += model.compute_metric(word,sememes,k=128)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(
                loss=loss.item(),
                r_at_8=recall_at_8,
                r_at_16=recall_at_16,
                r_at_32=recall_at_32,
                r_at_64=recall_at_64,
                r_at_128=recall_at_128,
            )

        dev_metric = validate(valid_iter, model, word_dict, sememe_dict,32)
        lr_scheduler.step(dev_metric.precision())
        if dev_metric > best_metric:
            best_metric = dev_metric
            print("New Best Metric: {}".format(dev_metric))
            print(' R@8: {}, R@16: {}, R@32: {}, R@64: {}, R@128:{}'.format(
                validate(valid_iter, model, word_dict, sememe_dict,8),
                validate(valid_iter, model, word_dict, sememe_dict,16),
                validate(valid_iter, model, word_dict, sememe_dict,32),
                validate(valid_iter, model, word_dict, sememe_dict,64),
                validate(valid_iter, model, word_dict, sememe_dict,128),
            ))
            torch.save(model.state_dict(), args.save_path)
            with open(args.save_path+'.sememe_vector','w') as f:
                for sym,vec in zip(sememe_dict.symbols,model.fc.weight.data):
                    f.write(sym + ' ' + ' '.join(map(str,vec.tolist())) + '\n')
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            break

    model.load_state_dict(torch.load(args.save_path))
    test_iter = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=12,
        collate_fn=SememePredictorDataset.collate_fn,
    )
    test_metric = validate(test_iter, model, word_dict, sememe_dict)
    print("Test FScore: {}".format(test_metric))


def validate(dataiter, model, word_dict, sememe_dict,K=32):
    model.eval()
    with torch.no_grad():
        # pred_labels = []
        # target_labels = []
        dev_metric = Accuracy()

        for i, batch in enumerate(dataiter):
            word = batch["word"].to(device)
            sememes = batch["sememes"].to(device)
            dev_metric += model.compute_metric(word, sememes, K)
    return dev_metric


def analyze_sememe(dataiter, model, word_dict, sememe_dict, embed_dict, fname):
    model.eval()
    with torch.no_grad(), open(fname, "w") as f:
        sememe_embeddings = model.fc.weight.data
        W, D = model.word_embed.weight.data.size()
        for i, sememe_vec in enumerate(sememe_embeddings):
            if i < sememe_dict.nspecial:
                continue
            score = F.cosine_similarity(
                sememe_vec.unsqueeze(0).expand(W, D), model.word_embed.weight.data
            ).squeeze()
            sorted, indices = score.sort(dim=-1, descending=True)
            f.write(sememe_dict[i] + ": ")
            for j in indices[:10]:
                f.write(word_dict[j] + " ")
            f.write("\n")


def analyze_word(dataiter, model, word_dict, sememe_dict, embed_dict, fname):
    model.eval()
    with torch.no_grad(), open(fname, "w") as f:
        for i, batch in enumerate(dataiter):
            word = batch["word"].to(device)
            sememes = batch["sememes"].to(device)
            logits = torch.sigmoid(model(word))

            for w, l, t in zip(word, logits, sememes):
                p = l > 0.5
                if p.sum() == 0:
                    _, indices = l.sort(dim=-1, descending=True)
                    pred_label = indices[:3]

                else:
                    pred_label, = np.where(p.cpu().numpy() == 1)
                target_label = t.cpu().numpy()

                f.write(word_dict[w] + "\n")
                f.write(
                    "Gold: {}".format(
                        " ".join(
                            [sememe_dict[x] for x in target_label.tolist() if x > 4]
                        )
                    )
                    + "\n"
                )
                f.write(
                    "Pred: {}".format(
                        " ".join([sememe_dict[x] for x in pred_label.tolist()])
                    )
                )
                f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", default="train")
    parser.add_argument("--data-path", default="./data/hownet")
    parser.add_argument(
        "--embed-path"
    )
    parser.add_argument(
        "--sememe-embed-path",
    )
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--embed-dim", default=300)
    parser.add_argument("--latent-K", default=2)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--epoch", default=100)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--save-path", default="best.pth")

    args = parser.parse_args()

    main(args)
