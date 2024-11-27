from torch.nn.modules import loss
from model.hash_model import DSPH as DSPH
import os
from torch.nn import functional
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import scipy.io as scio
from hhf import HHF
from .base import TrainBase
from model.optimization import BertAdam
from utils import get_args, calc_neighbor, cosine_similarity, euclidean_similarity
from utils.calc_utils import calc_map_k_matrix as calc_map_k
# from utils.calc_utils import calc_map_k
from utils.utils import HyP
from dataset.dataloader import dataloader
import json
import os
from ASL import AsymmetricLoss
import time
from copy import deepcopy

import torch
from loguru import logger
# pip install pytorch-metric-learning
from pytorch_metric_learning import distances, reducers
from torch.optim import Adam

from AdaTriplet.config import get_config
from AdaTriplet.losses import TripletCustomMarginLoss, LowerBoundLoss ,bit_var_loss
from AdaTriplet.methods import MetricLearningMethods
from AdaTriplet.miners.triplet_automargin_miner import TripletAutoParamsMiner
from AdaTriplet.networks import BackboneModel
import time
from losses import contrastive_jaccard

from utils.CPF_loss import CPF

from DSH import DSHLoss
from loss import RelaHashLoss
from relative_similarity import *
from alex import AlexNet
from timm.utils import AverageMeter
from pytorch_metric_learning import distances, reducers
from FAST_HPP import HouseHolder

class Trainer(TrainBase):

    def __init__(self,
                rank=1):
        args = get_args()
        super(Trainer, self).__init__(args, rank)
        self.logger.info("dataset len: {}".format(len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")

        self.logger.info("ViT+GPT!")
        HashModel = DSPH
        self.model = HashModel(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
                            writer=self.writer, logger=self.logger, is_train=self.args.is_train).to(self.rank)
        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(self.args.pretrained, map_location=f"cuda:{self.rank}"))
        
        self.model.float()


        self.rot = HouseHolder(dim=self.args.output_dim).to(self.rank)###
        # self.alex = AlexNet(hash_bit=self.args.output_dim).to(self.rank)
        self.optimizer = BertAdam([
                    {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
                    {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
                    {'params': self.model.text_hash.parameters(), 'lr': self.args.lr},
                    {'params': self.rot.parameters(), 'lr': self.args.clip_lr},
                    # {'params': self.alex.parameters(), 'lr': self.args.lr},
                    ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
                    b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
                    weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        self.cpf = CPF(embed_dim=self.args.output_dim, n_classes=self.args.numclass, device=1)


        self.optimizer_loss = torch.optim.Adam(params=self.cpf.parameters(), lr=1e-5)


        # self.rela_optimizer_loss = optim.Adam(self.relative_similarity.parameters(), lr=1e-5)
        self.total_time = 0

    def _init_dataset(self):
        self.config = get_config()
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset.")
        self.args.index_file = os.path.join("./dataset", self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join("./dataset", self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join("./dataset", self.args.dataset, self.args.label_file)
        train_data, query_data, retrieval_data = dataloader(captionFile=self.args.caption_file, 
                                        indexFile=self.args.index_file, 
                                        labelFile=self.args.label_file, 
                                        maxWords=self.args.max_words,
                                        imageResolution=self.args.resolution,
                                        query_num=self.args.query_num,
                                        train_num=self.args.train_num,
                                        seed=self.args.seed)
        self.train_labels = train_data.get_all_label().to(1)
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
        self.train_loader = DataLoader(
                dataset=train_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )
        self.query_loader = DataLoader(
                dataset=query_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )
        self.retrieval_loader = DataLoader(
                dataset=retrieval_data,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=True,
                shuffle=True
            )





    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(">>>>>> epochs: %d/%d"%(epoch, self.args.epochs))
        all_loss = 0

        ##
        distance = distances.CosineSimilarity()
        reducer = reducers.ThresholdReducer(low=0)
        mining_func = TripletAutoParamsMiner(
            distance=distance,
            margin_init=self.config.margin_m_loss,
            beta_init=self.config.margin_beta,
            type_of_triplets=self.config.type_of_triplets,
            k=self.config.k_param_automargin,
            k_n=self.config.k_n_param_autobeta,
            k_p=self.config.k_p_param_autobeta,
            mode=self.config.automargin_mode,
        )
        loss_matching_func = TripletCustomMarginLoss(margin=self.config.margin_m_loss, distance=distance, reducer=reducer)
        loss_id_func = LowerBoundLoss()
        # ##

        for iteration , (image, text, label, index) in enumerate(self.train_loader):
            start_time = time.time()
            self.global_step += 1
            image.float()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True).float()

            hash_img, hash_text = self.model(image, text)

            img_rot = F.normalize(self.rot(hash_img.T).T)###
            text_rot = F.normalize(self.rot(hash_text.T).T)###

            loss = self.cpf(hash_img, hash_text, label)

            #######################################################################################
            criterion = bit_var_loss()
            method = MetricLearningMethods(self.config, mining_func, loss_matching=loss_matching_func,
                                           loss_identity=loss_id_func)
            t_img_loss = method.calculate_total_loss(hash_img, label, epoch_id=epoch, batch_id=iteration)
            q_img_loss = criterion(img_rot)#####
            t_text_loss = method.calculate_total_loss(hash_text, label, epoch_id=epoch, batch_id=iteration)
            q_text_loss = criterion(text_rot)####
            lossLQ = t_text_loss + q_text_loss + q_img_loss + t_img_loss
            all_loss = loss + lossLQ
            ###############################################################################################
            # w = ((label.float() @ label.float().T) > 0).to(1)
            # scoresi = torch.nn.functional.normalize(hash_img) @ torch.nn.functional.normalize(hash_img).T
            # scorest = torch.nn.functional.normalize(hash_text) @ torch.nn.functional.normalize(hash_text).T
            # # scoresi = hash_img.float() @ hash_img.float().T
            # # scorest = hash_text.float() @ hash_text.float().T
            # ioss_context, ireg_loss = contrastive_jaccard(scoresi, w)
            # toss_context, treg_loss = contrastive_jaccard(scorest, w)
            # all_loss = loss + ioss_context + ireg_loss +toss_context + treg_loss
            #############################
            self.optimizer.zero_grad()
            self.optimizer_loss.zero_grad()


            all_loss.backward()

            self.optimizer.step()
            self.optimizer_loss.step()
            self.total_time += time.time() - start_time



        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}] loss: {all_loss.data / (len(self.train_loader))}, lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(self.optimizer.get_lr())))])}, time: {self.total_time}")

    def train(self):
        self.logger.info("Start train.")

        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            self.valid(epoch)
            self.save_model(epoch)

        self.logger.info(f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")


    def get_code(self, data_loader, length: int):

        img_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, self.args.output_dim, dtype=torch.float).to(self.rank)
        encoder_time = 0
        for image, text, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            image_hash = self.model.encode_image(image)
            # image_hash = self.alex(image)
            image_hash = torch.sign(image_hash)
            text_hash = self.model.encode_text(text)
            text_hash = torch.sign(text_hash)
            encoder_time = time.time() - start_encoder_time
            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data
        
        return img_buffer, text_buffer, encoder_time


    def test(self, mode_name="i2t"):
        if self.args.pretrained == "":
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")
        self.change_state(mode="valid")
        save_dir = os.path.join(self.args.save_dir, "11")
        os.makedirs(save_dir, exist_ok=True)
        query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.args.retrieval_num)
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, 5000, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, 5000, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, 5000, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, 5000, self.rank)
        # mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, 50).to(1)
        # mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, 50).to(1)
        # mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, 50).to(1)
        # mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, 50).to(1)
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(f">>>>>> MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}")

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir, str(self.args.outputdim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
        self.logger.info(">>>>>> save all data!")


    def valid(self, epoch):
        self.logger.info("Valid.")
        self.change_state(mode="valid")
        query_img, query_txt, q_encoder_time = self.get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt, r_encoder_time = self.get_code(self.retrieval_loader, self.args.retrieval_num)
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, 5000, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, 5000, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, 5000, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, 5000, self.rank)
        # mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels, self.retrieval_labels, 50).to(1)
        # mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels, self.retrieval_labels, 50).to(1)
        # mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels, self.retrieval_labels, 50).to(1)
        # mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels, self.retrieval_labels, 50).to(1)
        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t")
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            self.save_mat(query_img, query_txt, retrieval_img, retrieval_txt, mode_name="t2i")
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, \
                    MAX MAP(i->t): {self.max_mapi2t}, MAX MAP(t->i): {self.max_mapt2i}, query_encoder_time: {q_encoder_time}, retrieval_encoder_time: {r_encoder_time}")

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t"):

        save_dir = os.path.join(self.args.save_dir, "11")
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir, str(self.args.output_dim) + "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
        self.logger.info(f">>>>>> save best {mode_name} data!")


