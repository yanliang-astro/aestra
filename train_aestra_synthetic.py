#!/usr/bin/env python

import time, argparse, os
import numpy as np
import functools
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator
# allows one to run fp16_train.py from home directory
import sys;sys.path.insert(1, './')
from spender_model import SpectrumAutoencoder,NullRVEstimator
from synthetic_data import Synthetic
from util import mem_report
from functools import partial
from util import BatchedFilesDataset, load_batch
from torch.utils.data import DataLoader,Dataset
from torchinterp1d import Interp1d
from line_profiler import LineProfiler
from scipy.special import digamma

def corrcoef(tensor, rowvar=True, bias=False):
    """Estimate a corrcoef matrix (np.corrcoef)
    https://gist.github.com/ModarTensai/5ab449acba9df1a26c12060240773110
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    covmat = factor * tensor @ tensor.transpose(-1, -2).conj()
    std = torch.diag(covmat)**0.5
    covmat /= std[:, None]
    covmat /= std[None, :]
    return covmat
    
def avgdigamma(dist,radius):
    num_points = torch.count_nonzero(dist<(radius-1e-15),dim=0).double()
    return (torch.digamma(num_points)).mean()

def mutual_information(x0, y0, k=50, base=2):
    x = (x0-x0.mean(dim=0))/x0.std(dim=0)
    y = (y0-y0.mean())/y0.std()
    """Mutual information of x and y 
    """
    assert x.shape[0] == y.shape[0], "Arrays should have same length"
    assert y.shape[1] == 1,     "Single value function"
    assert k <= x.shape[0] - 1, "Set k smaller than num. samples - 1"
    # Find nearest neighbors in joint space, 
    points = torch.cat((x, y),dim=1)

    # Find nearest neighbors in joint space, p=inf means max-norm
    distmat = torch.abs(points[:,None]-points[None,:])
    dvec = torch.kthvalue(torch.amax(distmat,2),k+1,dim=1)[0]
    
    a, b, c, d = (
        avgdigamma(torch.amax(distmat[:,:,:-1],2),dvec),
        avgdigamma(distmat[:,:,-1],dvec),
        digamma(k),
        digamma(x.shape[0]),
    )
    return (-a - b + c + d) / np.log(base)

def prepare_train(seq,niter=100000):
    for d in seq:
        if not "iteration" in d:d["iteration"]=niter
        if not "encoder" in d:d.update({"encoder":d["data"]})
    return seq

def build_ladder(train_sequence):
    n_iter = sum([item['iteration'] for item in train_sequence])

    ladder = np.zeros(n_iter,dtype='int')
    n_start = 0
    for i,mode in enumerate(train_sequence):
        n_end = n_start+mode['iteration']
        ladder[n_start:n_end]= i
        n_start = n_end
    return ladder

def get_all_parameters(models,instruments):
    model_params = []
    # multiple encoders
    for model in models:
        model_params += model.encoder.parameters()
        model_params += model.rv_estimator.parameters()
    # 1 decoder
    model_params += model.decoder.parameters()
    dicts = [{'params':model_params}]

    n_parameters = sum([p.numel() for p in model_params if p.requires_grad])

    instr_params = []
    # instruments
    for inst in instruments:
        if inst==None:continue
        instr_params += inst.parameters()
        s = [p.numel() for p in inst.parameters()]
        #print("Adding %d parameters..."%sum(s))
    if instr_params != []:
        dicts.append({'params':instr_params,'lr': 1e-4})
        n_parameters += sum([p.numel() for p in instr_params if p.requires_grad])
        print("parameter dict:",dicts[1])
    return dicts,n_parameters

def consistency_loss(s, s_aug, individual=False, sigma_s=0.02):
    batch_size, s_size = s.shape
    ds = torch.sum((s_aug - s)**2/(sigma_s)**2,dim=1)/(s_size)
    cons_loss = torch.sigmoid(ds)-0.5 # zero = perfect alignment
    if individual:
        return cons_loss
    return cons_loss.sum()

def z_offset_loss(z_off, z_off_true, sigma_z=3.3e-9,individual=False):
    z_loss = ((z_off - z_off_true)/sigma_z)**2
    if individual:return z_loss
    return z_loss.sum()

def restframe_weight(model,instrument,xrange=[5372.52,5476.09],sn=300):
    x = model.decoder.wave_rest
    w = torch.zeros_like(x).float()
    w[(x>xrange[0])*(x<xrange[1])]=sn**2
    return w

def similarity_loss(x,slope=1.0,wid=5.0):
    sim_loss = torch.sigmoid(slope*x-wid/2)+torch.sigmoid(-slope*x-wid/2)
    return sim_loss

def similarity_restframe(instrument, model, s=None, slope=1.0, sigma_s=0.02,
                         individual=False, sn=300):
    _, s_size = s.shape
    device = s.device

    spec = model.decode(s)
    batch_size, spec_size = spec.shape

    # pairwise dissimilarity of spectra
    S = (spec[None,:,:] - spec[:,None,:])**2
    # dissimilarity of spectra
    # of order unity, larger for spectrum pairs with more comparable bins
    W = restframe_weight(model,instrument,sn=sn)
    spec_sim = (W * S).sum(-1) / W.count_nonzero()
    # dissimilarity of latents
    s_sim = ((s[None,:,:] - s[:,None,:])**2/sigma_s**2).sum(-1) / s_size

    x = s_sim-spec_sim
    sim_loss = similarity_loss(x,slope=slope)
    diag_mask = torch.diag(torch.ones(batch_size,device=device,dtype=bool))
    sim_loss[diag_mask] = 0
    if individual:
        return s_sim,spec_sim,sim_loss
    print("sigma_s:",sigma_s)
    print("s_sim:",s_sim,"spec_sim:",spec_sim)
    # total loss: sum over N^2 terms,
    # needs to have amplitude of N terms to compare to fidelity loss
    return sim_loss.sum() / batch_size

def _losses(model,
            instrument,
            batch,
            template=None,
            similarity=False,
            slope=0,
            sigma_s=1.0,
            fid=True,
            skipz=False,
            mi=False):

    spec, w, _, ID = batch

    if template==None: template = 0

    if fid: s = model.encode(spec-template)
    else: s = 0.0

    if skipz:
        rv = torch.zeros((len(spec),1),device=spec.device)
    else:rv =  model.estimate_rv(spec-template)

    z = (rv)/instrument.c

    fid_loss = sim_loss = flex_loss = 0

    if fid:
        y_act, _, spectrum_observed = model._forward(spec, w, s, z)
        fid_loss = model._loss(spec, w, spectrum_observed)
        flex_loss = slope*(y_act**2/1).sum()

    if similarity:
        sim_loss = similarity_restframe(instrument, model, s, slope=slope,sigma_s=sigma_s)
    if mi:flex_loss = mutual_information(s,rv)*spec.shape[0]

    return fid_loss, sim_loss, flex_loss, s, z

def get_losses(model,
               instrument,
               batch,
               template,
               aug_fct=None,
               similarity=True,
               consistency=True,
               flexibility=True,
               slope=0,
               sigma_s=2,
               zloss=True,
               skipfid=False,
               skipz=False
               ):

    zeropoint_loss = 0

    loss,sim_loss,flex_loss,s,z = _losses(model, instrument, batch, similarity=similarity, slope=slope, sigma_s=sigma_s, fid=not skipfid,skipz=skipz,template=template)

    if not skipz and (zloss or consistency):
        batch_copy = aug_fct(batch)
        fid_loss,_,_,s_,z_ = _losses(model, instrument,batch_copy, template=template,fid=False,skipz=skipz,similarity=False)
        z_off = z_ - z
        z_off_true = batch_copy[3]
        #flex_loss = fid_loss

    if not skipz and zloss:
        z_loss = z_offset_loss(z_off, z_off_true)
        print("z_loss:",z_loss.item(),
              "RV: %.2f, %.2f"%(Synthetic.c*z.min(),
                                Synthetic.c*z.max()))
    else: z_loss = 0

    if consistency and aug_fct is not None:
        cons_loss = consistency_loss(s, s_, sigma_s=sigma_s)
    else: cons_loss = 0

    if not flexibility: flex_loss = 0

    return loss, sim_loss, z_loss, cons_loss, zeropoint_loss, flex_loss


def checkpoint(accelerator, args, optimizer, scheduler, n_encoder, outfile, losses):
    unwrapped = [accelerator.unwrap_model(args_i).state_dict() for args_i in args]

    accelerator.save({
        "model": unwrapped,
        "losses": losses,
    }, outfile)
    return

def update_rv_estimator(rvfile, models, instruments, name='rv_estimator'):
    device = instruments[0].wave_obs.device
    rv_model = torch.load(rvfile, map_location=device)["model"][0]
    short = lambda key: key.split(name+".")[1]
    rv_model = {short(key):val for key,val in rv_model.items() if name in key}
    for model in models: model.rv_estimator.load_state_dict(rv_model)
    return models

def load_model(mainfile, models, instruments):
    device = instruments[0].wave_obs.device
    model_struct = torch.load(mainfile, map_location=device)

    for i, model in enumerate(models):
        model.load_state_dict(model_struct['model'][i], strict=False)

    losses = model_struct['losses']
    return models, losses

def get_data_loader(datadir, select="mockdata",which="train",batch_size=1024,shuffle=True,double=False):
    if which=="train":file = os.path.join(datadir,"%s.pkl"%select)
    else: file = os.path.join(datadir,"%s_valid.pkl"%select)
    print("file:",file, "double:",double)
    data = load_batch(file)
    if double:data = [item.double() for item in data]
    else: data[:3] = [item.float() for item in data[:3]]
    print("select:",select,"which:",which,"data:",data[0].shape)
    dataset = torch.utils.data.TensorDataset(*data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train(models,
          instruments,
          trainloaders,
          validloaders,
          template_data,
          n_epoch=200,
          outfile=None,
          losses=None,
          verbose=False,
          lr=1e-4,
          n_batch=50,
          aug_fcts=None,
          similarity=True,
          consistency=True,
          flexibility=True,
          skipfid=False,
          skipz=False
          ):

    n_encoder = len(models)
    model_parameters, n_parameters = get_all_parameters(models,instruments)

    if verbose:
        print("model parameters:", n_parameters)
        mem_report()

    ladder = build_ladder(train_sequence)
    optimizer = optim.Adam(model_parameters, lr=lr, eps=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr,
                                              total_steps=n_epoch)

    #accelerator = Accelerator(mixed_precision='fp16')
    accelerator = Accelerator()
    models = [accelerator.prepare(model) for model in models]
    instruments = [accelerator.prepare(instrument) for instrument in instruments]
    trainloaders = [accelerator.prepare(loader) for loader in trainloaders]
    validloaders = [accelerator.prepare(loader) for loader in validloaders]
    template_data = [accelerator.prepare(item) for item in template_data]
    optimizer = accelerator.prepare(optimizer)

    template_data=template_data[0].to(instruments[0].wave_obs.device)
    # define losses to track
    n_loss = 6
    epoch = 0
    if losses is None:
        detailed_loss = np.zeros((2, n_encoder, n_epoch, n_loss))
    else:
        try:
            non_zero = np.sum(losses[0][0],axis=1)>0
            losses = losses[:,:,non_zero,:]

            epoch = len(losses[0][0])

            n_epoch += epoch
            detailed_loss = np.zeros((2, n_encoder, n_epoch, n_loss))
            detailed_loss[:, :, :epoch, :] = losses

            if verbose:
                losses = tuple(detailed_loss[0, :, epoch-1, :])
                vlosses = tuple(detailed_loss[1, :, epoch-1, :])
                print(f'====> Epoch: {epoch-1}')
                print('TRAINING Losses:', losses)
                print('VALIDATION Losses:', vlosses)
        except: # OK if losses are empty
            print("loss empty...")
            pass

    if outfile is None:
        outfile = "checkpoint.pt"

    for epoch_ in range(epoch, n_epoch):

        mode = train_sequence[ladder[epoch_ - epoch]]

        # turn on/off model decoder
        for p in models[0].decoder.parameters():
            p.requires_grad = mode['decoder']
        models[0].decoder.spec_rest.requires_grad = mode['spec_rest']

        slope = ANNEAL_SCHEDULE[(epoch_ - epoch)%len(ANNEAL_SCHEDULE)]
        if n_epoch-epoch_<=10: slope=0 # turn off similarity
        
        if verbose and similarity:
            print("similarity info:",slope)

        for which in range(n_encoder):

            # turn on/off encoder
            print("Encoder:",mode['encoder'][which])
            for p in models[which].encoder.parameters():
                p.requires_grad = mode['encoder'][which]
            # turn on/off rv_estimator
            print("RV estimator:",mode['rv'][which])
            for p in models[which].rv_estimator.parameters():
                p.requires_grad = mode['rv'][which]

            # optional: training on single dataset
            if not mode['data'][which]:
                continue

            models[which].train()
            instruments[which].train()

            n_sample = 0
            for k, batch in enumerate(trainloaders[which]):
                batch_size = len(batch[0])
                losses = get_losses(
                    models[which],
                    instruments[which],
                    batch,
                    template_data,
                    aug_fct=aug_fcts[which],
                    similarity=similarity,
                    consistency=consistency,
                    flexibility=flexibility,
                    slope=slope,
                    skipfid=skipfid,
                    skipz=skipz
                )
                # sum up all losses
                loss = functools.reduce(lambda a, b: a+b , losses)
                accelerator.backward(loss)
                # clip gradients: stabilizes training with similarity
                accelerator.clip_grad_norm_(model_parameters[0]['params'], 1.0)
                # once per batch
                optimizer.step()
                optimizer.zero_grad()

                # logging: training
                detailed_loss[0][which][epoch_] += tuple( l.item() if hasattr(l, 'item') else 0 for l in losses )
                n_sample += batch_size

                # stop after n_batch
                if n_batch is not None and k == n_batch - 1:
                    break
            detailed_loss[0][which][epoch_] /= n_sample

        scheduler.step()
        '''
        with torch.no_grad():
            for which in range(n_encoder):
                models[which].eval()
                instruments[which].eval()

                n_sample = 0
                for k, batch in enumerate(validloaders[which]):
                    batch_size = len(batch[0])
                    losses = get_losses(
                        models[which],
                        instruments[which],
                        batch,
                        template_data,
                        aug_fct=aug_fcts[which],
                        similarity=similarity,
                        consistency=consistency,
                        flexibility=flexibility,
                        slope=slope,
                        skipfid=skipfid,
                        skipz=skipz
                    )
                    # logging: validation
                    detailed_loss[1][which][epoch_] += tuple( l.item() if hasattr(l, 'item') else 0 for l in losses )
                    n_sample += batch_size

                    # stop after n_batch
                    if n_batch is not None and k == n_batch - 1:
                        break

                detailed_loss[1][which][epoch_] /= n_sample
        '''
        if verbose:
            #mem_report()
            losses = tuple(detailed_loss[0, :, epoch_, :])
            vlosses = tuple(detailed_loss[1, :, epoch_, :])
            print('====> Epoch: %i'%(epoch_))
            print('TRAINING Losses:', losses)
            print('VALIDATION Losses:', vlosses)

        if epoch_ % 20 == 0 or epoch_ == n_epoch - 1:
            args = models
            checkpoint(accelerator, args, optimizer, scheduler, n_encoder, outfile, detailed_loss)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="dataset name")
    parser.add_argument("dir", help="data file directory")
    parser.add_argument("outfile", help="output file name")
    parser.add_argument("-n", "--latents", help="latent dimensionality", type=int, default=2)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=512)
    parser.add_argument("-l", "--batch_number", help="number of batches per epoch", type=int, default=None)
    parser.add_argument("-r", "--rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-z", "--rv_file", help="rv estimator", type=str, default="None")
    parser.add_argument("-it", "--iteration", help="number of interation", type=int, default=100000)
    parser.add_argument("-s", "--similarity", help="add similarity loss", action="store_true")
    parser.add_argument("-skipfid", "--skipfid", help="skip fidelity loss", action="store_true",default=False)
    parser.add_argument("-skipz", "--skipz", help="skip rv loss", action="store_true",default=False)
    parser.add_argument("-c", "--consistency", help="add consistency loss", action="store_true")
    parser.add_argument("-d", "--double", help="double precision", action="store_true",default=False)
    parser.add_argument("-init", "--init", help="initialize restframe", action="store_true",default=False)
    parser.add_argument("-f", "--flexibility", help="constrian model flexibility", action="store_true",default=False)
    parser.add_argument("-C", "--clobber", help="continue training of existing model", action="store_true")
    parser.add_argument("-v", "--verbose", help="verbose printing", action="store_true")
    args = parser.parse_args()

    # define instruments
    instruments = [ Synthetic() ]
    n_encoder = len(instruments)

    # restframe wavelength for reconstructed spectra
    #lmbda_min = 4999.0;lmbda_max = 5011.0;bins = 1200
    wave_rest = Synthetic.wave_rest

    # data loaders
    trainloaders = [ inst.get_data_loader(args.dir, select=args.data, which="train",
                     batch_size=args.batch_size) for inst in instruments ]
    validloaders = [ inst.get_data_loader(args.dir, select=args.data, which="valid",
                     batch_size=args.batch_size) for inst in instruments ]

    template_data = load_batch("%s%s-template.pkl"%(args.dir,args.data))

    if args.init: init_restframe = load_batch("%s%s-rest.pkl"%(args.dir,args.data))[0]
    else: 
        init_restframe = Interp1d()(instruments[0].wave_obs, template_data[0], wave_rest)

    if args.double:
        template_data = [item.double() for item in template_data]
        if args.init: init_restframe = init_restframe.double()

    # get augmentation function
    aug_fcts = [ Synthetic.augment_spectra ]

    # define training sequence
    FULL = {"data":[True],"encoder":[True],"rv":[True],
            "decoder":True,"spec_rest":True}
    train_sequence = prepare_train([FULL],niter=args.iteration)

    annealing_step = 100
    ANNEAL_SCHEDULE = np.linspace(0.0,2.0,annealing_step)
    if args.verbose and args.similarity:
        print("similarity_slope:",len(ANNEAL_SCHEDULE),ANNEAL_SCHEDULE)

    # define and train the model
    n_hidden = (64, 256, 1024)
    models = [ SpectrumAutoencoder(instrument,
                                   wave_rest,
                                   spec_rest=init_restframe,
                                   n_latent=args.latents,
                                   n_hidden=n_hidden,
                                   n_aux=0,
                                   normalize=False)
              for instrument in instruments ]
    print("RVEstimator:",models[0].rv_estimator)

    # use same decoder
    if n_encoder==2:models[1].decoder = models[0].decoder
    if args.double:[model.double() for model in models]
    n_epoch = sum([item['iteration'] for item in train_sequence])
    init_t = time.time()
    if args.verbose:
        print("torch.cuda.device_count():",torch.cuda.device_count())
        print (f"--- Model {args.outfile} ---")

    # check if outfile already exists, continue only of -c is set
    if os.path.isfile(args.outfile) and not args.clobber:
        raise SystemExit("\nOutfile exists! Set option -C to continue training.")
    losses = None
    if os.path.isfile(args.outfile):
        if args.verbose:
            print("\nLoading file %s"%args.outfile)
        models, losses = load_model(args.outfile, models, instruments)

    if os.path.isfile(args.rv_file):
        if args.verbose:
            print("\nUpdating RV estimator based on file %s"%args.rv_file)
        models = update_rv_estimator(args.rv_file, models, instruments)

    profiler = LineProfiler()
    profiler.add_function(partial)
    profiler.add_function(load_batch)
    lpWrapper = profiler(train)
    lpWrapper(models, instruments, trainloaders, validloaders, template_data, n_epoch=n_epoch,
          n_batch=args.batch_number, lr=args.rate, aug_fcts=aug_fcts, similarity=args.similarity, consistency=args.consistency, flexibility=args.flexibility, skipfid=args.skipfid,skipz=args.skipz,outfile=args.outfile, losses=losses, verbose=args.verbose)
    
    profiler.print_stats()
    #train(models, instruments, trainloaders, validloaders, template_data, n_epoch=n_epoch,n_batch=args.batch_number, lr=args.rate, aug_fcts=aug_fcts, similarity=args.similarity, consistency=args.consistency, outfile=args.outfile, losses=losses, verbose=args.verbose)

    if args.verbose:
        print("--- %s seconds ---" % (time.time()-init_t))
