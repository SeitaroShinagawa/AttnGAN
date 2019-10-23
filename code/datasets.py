from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def prepare_data(data, use_attribute=False, use_embedding=False):
    imgs, captions, captions_lens, class_ids, keys = data[:5]
    if use_attribute and use_embedding:
        assert len(data[5:]) == 2
        embeddings, attributes = data[5:]
    elif use_attribute:
        assert len(data[5:]) == 1
        attributes = data[5]
    elif use_embedding:
        assert len(data[5:]) == 1
        embeddings = data[5]

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    wrong_imgs = []
    wrong_indices = torch.randperm(len(sorted_cap_indices))
    for i in range(len(imgs)):
        r_imgs = imgs[i][sorted_cap_indices] #sort by caption length order
        w_imgs = imgs[i][wrong_indices] #sort by caption length order
        if cfg.CUDA:
            real_imgs.append(Variable(r_imgs).cuda())
            wrong_imgs.append(Variable(w_imgs).cuda())
        else:
            real_imgs.append(Variable(r_imgs))
            wrong_imgs.append(Variable(w_imgs))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        if use_embedding:
            embeddings = embeddings[sorted_cap_indices].cuda()
        if use_attribute:
            attributes = attributes[sorted_cap_indices].float().cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
        if use_embedding:
            embeddings = embeddings[sorted_cap_indices]
        if use_attribute:
            attributes = attributes[sorted_cap_indices].float()
    
    out_list = [real_imgs, wrong_imgs, captions, sorted_cap_lens, class_ids, keys]
    if use_attribute:
        out_list.append(attributes)
    if use_embedding:
        out_list.append(embeddings)
    
    return out_list


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                #re_img = transforms.Scale(imsize[i])(img)
                re_img = transforms.Resize(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 base_size=64,
                 transform=None, target_transform=None,
                 use_attribute=False, use_embedding=False):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.specific_sent_ix = None

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        all_filenames = self.load_all_filenames()
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox(all_filenames)
        else:
            self.bbox = None
        
        if self.use_attribute:
            self.attributes, self.all_filename_id_dict = self.load_att(all_filenames)
            #self.att_conf = self.load_att_conf()

        self.split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        if self.use_embedding:
            self.embeddings = self.load_embedding(self.split_dir, 'myembedding')

        self.class_id = self.load_class_id(self.split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_all_filenames(self):
        data_dir = self.data_dir
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        all_filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(all_filenames), all_filenames[0])
        return all_filenames

    def load_att(self, all_filenames):
        data_dir = self.data_dir
        att_path = os.path.join(data_dir, 'CUB_200_2011/attributes/birds_att.npy')
        all_filename_id_dict = {filename:index for index, filename in enumerate(all_filenames)}
        return np.load(att_path), all_filename_id_dict
    
    def load_att_conf(self):
        conf_path = 'birds_att_conf.npy'
        conf_array = np.load(conf_path)
        return conf_array.astype("f")

    def load_bbox(self, all_filenames):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        
        filename_bbox = {img_file[:-4]: [] for img_file in all_filenames}
        numImgs = len(all_filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = all_filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r", encoding="latin1") as f:
                #captions = f.read().decode('utf8').split('\n')
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def specify_caption_index(index):
        self.specific_sent_ix = index
        print("specified sentence index = ", self.specific_sent_ix)
    
    def load_embedding(self, data_dir, embedding_type='myembedding'):
        if embedding_type == 'myembedding':
            embedding_filename = '/Myembedding.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            #embeddings = pickle.load(f, encoding="latin1")
            print('embeddings loaded')
        return embeddings


    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="latin1")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len


    def __getitem__(self, index):
        
        key = self.filenames[index]
        cls_id = self.class_id[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        
        # Get image
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        # Select a sentence
        if self.specific_sent_ix is not None:
            sent_ix = self.specific_sent_ix 
        else:
            sent_ix = random.randint(0, self.embeddings_num)
        
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)


        if self.use_attribute:
            attribute = self.attributes[self.all_filename_id_dict[key+".jpg"]] # (1,312) array
            #att_conf = self.att_conf[self.all_filename_id_dict[key+".jpg"]]

        if self.use_embedding: 
            embedding = self.embeddings[index, sent_ix, :]
        
        if self.use_attribute and self.use_embedding: 
            return imgs, caps, cap_len, cls_id, key, attribute, embedding
        elif self.use_attribute: 
            return imgs, caps, cap_len, cls_id, key, attribute
        elif self.use_embedding: 
            return imgs, caps, cap_len, cls_id, key, embedding
        else:
            return imgs, caps, cap_len, cls_id, key


    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    # python datasets.py --cfg cfg/bird_embedding.yml
    from main import parse_args
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    args.manualSeed = 100
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    
    dataset_train = TextDataset(cfg.DATA_DIR, 'train',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset_train
    dataset_test = TextDataset(cfg.DATA_DIR, 'test',
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset_test

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=False, shuffle=False, num_workers=int(cfg.WORKERS))
    
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=False, shuffle=False, num_workers=int(cfg.WORKERS))

	# Build text encoder    
    text_encoder = \
        RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = \
        torch.load(cfg.TRAIN.NET_E,
                   map_location=lambda storage, loc: storage) #load text encoder
    text_encoder.load_state_dict(state_dict)
    for p in text_encoder.parameters():
        p.requires_grad = False
    print('Load text encoder from:', cfg.TRAIN.NET_E)
    text_encoder.eval()

    if cfg.CUDA:
        text_encoder = text_encoder.cuda()

	def create_embeddings(dataloader, text_encoder):
		emb_dict = {}
        for sent_ix in range(cfg.TEXT.CAPTIONS_PER_IMAGE):
    	    dataloader.dataset.specify_caption_index(sent_ix) 
            for data in dataloader:
			    real_imgs, wrong_imgs, captions, sorted_cap_lens,
				    class_ids, keys = prepare_data(data)

                hidden = text_encoder.init_hidden(batch_size) 
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                for i, key in enumerate(keys):
                    emb_elements = { "words": words_embs[i],
                                    "sentence": sent_emb[i],
                                    "cap_len": sorted_cap_lens[i] }
                    if key not in emb_dict:
                        emb_dict[key] = [emb_elements]
                    else:
                        emb_dict[key].append(emb_elements)
		return emb_dict

	train_emb_dict = create_embeddings(dataloader_train, text_encoder) 
	test_emb_dict = create_embeddings(dataloader_test, text_encoder)
    def save_emb(path, emb_dict):
        with open(path, "wb") as f:
            pickle.dump(emb_dict, f)

    save_emb("embedding_train.pickle", train_emb_dict)
    save_emb("embedding_test.pickle", test_emb_dict)
