""" Works with pytorch 0.4.0 """

from .core import *
from .data_utils import pad_sequences, minibatches, get_chunks
from .crf import CRF
from .general_utils import Progbar
from torch.optim.lr_scheduler import StepLR

if os.name == "posix": from allennlp.modules.elmo import Elmo, batch_to_ids # AllenNLP is currently only supported on linux


class NERLearner(object):
    """
    NERLearner class that encapsulates a pytorch nn.Module model and ModelData class
    Contains methods for training a testing the model
    """
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.logger = self.config.logger
        self.model = model
        self.model_path = config.dir_model
        self.use_elmo = config.use_elmo


        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

        self.criterion = CRF(self.config.ntags)
        self.optimizer = optim.Adam(self.model.parameters())


        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0)


        if USE_GPU:
            self.use_cuda = True
            self.logger.info("GPU found.")
            self.model = model.cuda()
            self.criterion = self.criterion.cuda()
            if self.use_elmo:
                self.elmo = self.elmo.cuda()
                print("Moved elmo to cuda")
        else:
            self.model = model.cpu()
            self.use_cuda = False
            self.logger.info("No GPU found.")

    def get_model_path(self, name):
        return os.path.join(self.model_path,name)+'.h5'


    def save(self, name=None):
        if not name:
            name = self.config.ner_model_path
        save_model(self.model, self.get_model_path(name))
        self.logger.info(f"Saved model at {self.get_model_path(name)}")

    def load(self, fn=None):
        if not fn: fn = self.config.ner_model_path
        fn = self.get_model_path(fn)
        load_ner_model(self.model, fn, strict=True)
        self.logger.info(f"Loaded model from {fn}")

    def batch_iter(self, train, batch_size, return_lengths=False, shuffle=False, sorter=False):
        """
        Builds a generator from the given dataloader to be fed into the model

        Args:
            train: DataLoader
            batch_size: size of each batch
            return_lengths: if True, generator returns a list of sequence lengths for each
                            sample in the batch
                            ie. sequence_lengths = [8,7,4,3]
            shuffle: if True, shuffles the data for each epoch
            sorter: if True, uses a sorter to shuffle the data

        Returns:
            nbatches: (int) number of batches
            data_generator: batch generator yielding
                                dict inputs:{'word_ids' : np.array([[padded word_ids in sent1], ...])
                                             'char_ids': np.array([[[padded char_ids in word1_sent1], ...],
                                                                    [padded char_ids in word1_sent2], ...],
                                                                    ...])}
                                labels: np.array([[padded label_ids in sent1], ...])
                                sequence_lengths: list([len(sent1), len(sent2), ...])

        """
        nbatches = (len(train) + batch_size - 1) // batch_size
        print('Length of dataset: ', len(train))
        print('No of batches: ',nbatches)

        def data_generator():
            while True:
                if shuffle: train.shuffle()
                elif sorter==True and train.sorter: train.sort()

                for i, (words, labels) in enumerate(minibatches(train, batch_size)):


                    word_ids, sequence_lengths = pad_sequences(words, 0)
                    #word_ids = [bs,sl]

                    word_ids = words
                    #word_ids = [bs,sl]

                    if labels:
                        labels, _ = pad_sequences(labels, 0)
                        # if categorical
                        ## labels = [to_categorical(label, num_classes=len(train.tag_itos)) for label in labels]

                    # build dictionary
                    inputs = {
                        "word_ids": np.asarray(word_ids)
                    }

                    if return_lengths:
                        yield(inputs, np.asarray(labels), sequence_lengths)

                    else:
                        yield (inputs, np.asarray(labels))

        return (nbatches, data_generator())


    def fit(self, train, dev=None, epochs=None):
        """
        Fits the model to the training dataset and evaluates on the validation set.
        Saves the model to disk
        """
        if not epochs:
            epochs = self.config.nepochs
        batch_size = self.config.batch_size
        #print('batch_size: ',batch_size)

        nbatches_train, train_generator = self.batch_iter(train, batch_size,
                                                          return_lengths=True)
        if dev:
            nbatches_dev, dev_generator = self.batch_iter(dev, batch_size,
                                                      return_lengths=True)

        scheduler = StepLR(self.optimizer, step_size=1, gamma=self.config.lr_decay)

        self.logger.info("Training Model")

        f1s = []

        for epoch in range(epochs):
            scheduler.step()
            self.train(epoch, nbatches_train, train_generator)

            if dev:
                f1 = self.test(nbatches_dev, dev_generator)

            # Early stopping
            if len(f1s) > 0:
                if f1 < max(f1s[max(-self.config.nepoch_no_imprv, -len(f1s)):]): #if sum([f1 > f1s[max(-i, -len(f1s))] for i in range(1,self.config.nepoch_no_imprv+1)]) == 0:
                    print("No improvement in the last 3 epochs. Stopping training")
                    break
            else:
                f1s.append(f1)


        self.save(self.config.ner_model_path)


    def train(self, epoch, nbatches_train, train_generator):
        self.logger.info('\nEpoch: %d' % epoch)
        self.model.train()
        if not self.use_elmo: self.model.emb.weight.requires_grad = False

        train_loss = 0
        correct = 0
        total = 0
        total_step = None

        prog = Progbar(target=nbatches_train)

        for batch_idx, (inputs, targets, sequence_lengths) in enumerate(train_generator):

            if batch_idx == nbatches_train: break
            if inputs['word_ids'].shape[0] == 1:
                self.logger.info('Skipping batch of size=1')
                continue

            #print('inputs: ',inputs['word_ids'])
            #print('targets: ',targets.shape)
            #targets [bs,sl]

            total_step = batch_idx
            targets = T(targets, cuda=self.use_cuda).transpose(0,1).contiguous()
            #print('targets: ',targets.size())
            #targets[sl,bs]
            self.optimizer.zero_grad()

            sentences = inputs['word_ids']
            #print('sentences: ',sentences[0])
            #sentences = ["EU", "Union", "decides", "to", "cancel"]
            #sentences: list of list of [size batch size, sl]
            character_ids = batch_to_ids(sentences)
            #print('character_ids: ',character_ids.size())
            #character_ids = [bs,sl,50] = [5,31,50]
            if self.use_cuda:
                character_ids = character_ids.cuda()

            embeddings = self.elmo(character_ids)
            word_input = embeddings['elmo_representations'][0]
            #print('word_input: ',word_input.size())
            #word_input [bs,sl,1024]
            word_input, targets = Variable(word_input, requires_grad=False), \
                                  Variable(targets)
            #print('targets: ',targets.size())
            #targets = [sl,bs]
            inputs = (word_input)
            #print('inputs: ',inputs.size())
            #inputs = [bs,sl,1024]


            outputs = self.model(inputs)
            #print('outputs: ',outputs.size())
            #outputs = [sl,bs,n_tags]

            # Create mask

            mask = Variable(embeddings['mask'].transpose(0,1))
            #print('mask: ',mask.size())
            #mask [sl,bs]
            if self.use_cuda:
                mask = mask.cuda()

            # Get CRF Loss
            loss = -1*self.criterion(outputs, targets, mask=mask)
            #print(loss) #loss value tensor
            #print('loss: ',loss.size())
            loss.backward()
            self.optimizer.step()

            # Callbacks
            train_loss += loss.item()
            predictions = self.criterion.decode(outputs, mask=mask)
            #print('predictions: {}, {}'.format(len(predictions), len(predictions[0])))
            #print(predictions)
            #predictions = [5,9]
            masked_targets = mask_targets(targets, sequence_lengths)
            #print('masked_targets: ',masked_targets)
            #print('masked_targets: {}, {}'.format(len(masked_targets),len(masked_targets[0])))

            t_ = mask.type(torch.LongTensor).sum().item()
            total += t_
            c_ = sum([1 if p[i] == mt[i] else 0 for p, mt in zip(predictions, masked_targets) for i in range(len(p))])
            correct += c_

            prog.update(batch_idx + 1, values=[("train loss", loss.item())], exact=[("Accuracy", 100*c_/t_)])

        self.logger.info("Train Loss: %.3f, Train Accuracy: %.3f%% (%d/%d)" %(train_loss/(total_step+1), 100.*correct/total, correct, total) )


    def test(self, nbatches_val, val_generator):
        self.model.eval()
        accs = []
        test_loss = 0
        correct_preds = 0
        total_correct = 0
        total_preds = 0
        total_step = None

        for batch_idx, (inputs, targets, sequence_lengths) in enumerate(val_generator):
            if batch_idx == nbatches_val: break
            if inputs['word_ids'].shape[0] == 1:
                self.logger.info('Skipping batch of size=1')
                continue

            total_step = batch_idx
            targets = T(targets, cuda=self.use_cuda).transpose(0,1).contiguous()

            sentences = inputs['word_ids']
            character_ids = batch_to_ids(sentences)
            if self.use_cuda:
                character_ids = character_ids.cuda()
            embeddings = self.elmo(character_ids)
            word_input = embeddings['elmo_representations'][1]
            word_input, targets = Variable(word_input, requires_grad=False), \
                                  Variable(targets)
            inputs = (word_input)

            outputs = self.model(inputs)

            # Create mask
            mask = Variable(embeddings['mask'].transpose(0,1))
            if self.use_cuda:
                mask = mask.cuda()


            # Get CRF Loss
            loss = -1*self.criterion(outputs, targets, mask=mask)

            # Callbacks
            test_loss += loss.item()
            predictions = self.criterion.decode(outputs, mask=mask)
            masked_targets = mask_targets(targets, sequence_lengths)

            #predictions = [bs,sl]
            #masked_targets = [bs,sl]

            for lab, lab_pred in zip(masked_targets, predictions):
                #lab: single sent : list of target idx
                #lab_pred: list of pred idx

                accs    += [1 if a==b else 0 for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        self.logger.info("Val Loss : %.3f, Val Accuracy: %.3f%%, Val F1: %.3f%%" %(test_loss/(total_step+1), 100*acc, 100*f1))
        return 100*f1

    def evaluate(self,test):
        batch_size = self.config.batch_size
        nbatches_test, test_generator = self.batch_iter(test, batch_size,
                                                        return_lengths=True)
        self.logger.info('Evaluating on test set')
        self.test(nbatches_test, test_generator)

    def predict_batch(self, words):
        self.model.eval()
        if len(words) == 1:
            mult = np.ones(2).reshape(2, 1).astype(int)

        sentences = words
        sequence_lengths = [len(word) for word in words]
        character_ids = batch_to_ids(sentences)
        #character_ids = [1,sl,50]
        if self.use_cuda:
            character_ids = character_ids.cuda()
        embeddings = self.elmo(character_ids)
        word_input = embeddings['elmo_representations'][1]
        word_input = Variable(word_input, requires_grad=False)
        print('word_input: ',word_input.size())
        #word_input = [1,sl,1024] = [1,6,1024]

        """
        if len(words) == 1:
            word_input = ((mult*word_input.transpose(0,1)).transpose(0,1).contiguous()).type(torch.FloatTensor)
        """

        word_input = T(word_input, cuda=self.use_cuda)
        print('word_input: ',word_input.size())
        inputs = (word_input)
        print('inputs: ',inputs.size())



        outputs = self.model(inputs)

        predictions = self.criterion.decode(outputs)
        print('predictions: {}, {}'.format(len(predictions),len(predictions[0])))
        #predictions = [1,sl,n_tags]

        predictions = [p[:i] for p, i in zip(predictions, sequence_lengths)]

        return predictions

    def predict(self, sentences):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        nlp = spacy.load('en_core_web_sm')
        if type(sentences)!='str':
            sentences = " ".join(sentences)

        #print(sentences)
        doc = nlp(sentences)
        words_raw = [[token.text for token in sent] for sent in doc.sents]

        words = words_raw
        #words: list of list of words

        pred_ids = self.predict_batch(words)
        preds = [[self.idx_to_tag[idx.item() if isinstance(idx, torch.Tensor) else idx]  for idx in s] for s in pred_ids]

        return preds


def create_mask(sequence_lengths, targets, cuda, batch_first=False):
    """ Creates binary mask """
    mask = Variable(torch.ones(targets.size()).type(torch.ByteTensor))
    if cuda: mask = mask.cuda()

    for i,l in enumerate(sequence_lengths):
        if batch_first:
            if l < targets.size(1):
                mask.data[i, l:] = 0
        else:
            if l < targets.size(0):
                mask.data[l:, i] = 0

    return mask


def mask_targets(targets, sequence_lengths, batch_first=False):
    """ Masks the targets """
    if not batch_first:
         targets = targets.transpose(0,1)
    t = []
    for l, p in zip(targets,sequence_lengths):
        t.append(l[:p].data.tolist())
    return t
