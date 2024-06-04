from . import __version__
header = """\n
---------------------------------------------
  Remove Cysteines (v%s)
  CKT Lab -- http://ckinzthompson.github.io
---------------------------------------------
"""%(__version__)

import os
import sys
import esm
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="esm")

ESM_models = ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"]

devices = []
if torch.cuda.is_available():
	devices.append('cuda')
if torch.backends.mps.is_available():
	devices.append('mps')
devices.append('cpu')

letters = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']
letterids = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

def just_the_model(model_name,model_directory=None):
	if not model_directory is None:
		if not os.path.exists(model_directory):
			os.mkdir(model_directory)
		if os.path.exists(model_directory): ## a second check to make sure it got made...
			torch.hub.set_dir(model_directory)
	url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
	model_data = esm.pretrained.load_hub_workaround(url)
	return esm.pretrained.load_model_and_alphabet_core(model_name, model_data, None)

def pll_fxn(x,a,k,n):
	## Power law
	return a*(1-(k*(-x))**n)

def _minfxn(theta,x,y):
	a,k,n = theta
	rmsd = np.sqrt(np.mean(np.square(y-pll_fxn(x,a,k,n,))))
	return rmsd

def fit_pll(x,y,init_a=.01,init_k=100000.,init_n=.5,maxiters=10):
	guess = np.array((init_a,init_k,init_n))
	out = minimize(_minfxn,x0=guess,args=(x,y),method='Nelder-Mead')
	for _ in range(maxiters-1):
		out = minimize(_minfxn,x0=out.x,args=(x,y),method='Nelder-Mead')
	return out.success,out.x

def _optimize_maskless_scoring(lnp_wt,lnp_mask):
	#### Does the actual fitting

	success,theta = fit_pll(lnp_wt,lnp_mask) ## trying to predict lnp_mask
	a,k,n = theta

	print("Successful Fitting?",success)
	print('a=%.3f, k=%.3f, n=%.3f'%(a,k,n))

	## make plot
	xmin = lnp_wt.min() * 1.05
	ymin = lnp_mask.min() * 1.05

	xx = np.linspace(xmin,0,1000)
	yy = pll_fxn(xx,a,k,n)

	fig,ax = plt.subplots(1,figsize=(8,6))
	
	ax.plot(lnp_wt,lnp_mask,'o',color='tab:blue',alpha=.5)
	ax.plot(xx,yy,color='tab:red')
	ax.set_xlim(xmin,0)
	ax.set_ylim(ymin,0)
	ax.set_xlabel(r'$\ln P\left(x_i\mid x_i, x_{\backslash i} \right)$')
	ax.set_ylabel(r'$PLL(x_i) \equiv \ln P\left(x_i\mid m_i, x_{\backslash i} \right)$')
	ax.set_title(r'$\theta$: a=%.3f, k=%.1f, n=%.3f'%(a,k,n))
	fig.tight_layout()

	return theta,fig,ax

def logits_to_pppl_approx(data,logits,alphabet,theta):
	### logits should be (len(seqs),nseq+2,33) --> i.e., (num seq,seq len,AAs)
	### data should be [('name',seq1),('name 2',seq2),...]
	pppl_approx = np.zeros(len(data))
	
	for ind in range(len(data)):
		seq = data[ind][1]
		pmask = softmax(logits[ind,1:-1],axis=1) ## keep in case of normalization...
		targets = np.array([alphabet.tok_to_idx[seq[i]] for i in range(len(seq))])
		lnp_wt = np.log(np.array([pmask[i,targets[i]] for i in range(len(seq))]))
		lnp_wt2mask = pll_fxn(lnp_wt,*theta)
		pppl_approx[ind] = np.exp(-np.mean(lnp_wt2mask))
	return pppl_approx

def get_target_indices(seq,target):
	#### Find indices of target AA in sequence
	return np.array([index for index in range(len(seq)) if seq[index] == target]) 

def clean_sequence(seq):
	#### cleans and uppercases sequence
	return ''.join([si.upper() for si in seq if si.upper() in letters]) ## clean sequence


def embed_sequences(data,model,alphabet,device): 
	#### This function takes data list, [('name',seq1),('name 2',seq2),...], and applies the forward model
	batch_converter = alphabet.get_batch_converter()
	batch_labels, batch_strs, batch_tokens = batch_converter(data)
	batch_tokens = batch_tokens.to(device)
	with torch.no_grad():
		results = model(batch_tokens, repr_layers=[model.num_layers,], return_contacts=False)
	reps = results['representations'][model.num_layers].cpu().numpy().copy()
	logits = results['logits'].cpu().numpy().copy()
	return reps,logits

def generate_pointmutants(seq,index):
	#### Generates a data list [('name',seq1),('name 2',seq2),...] of all possible 20 point mutations at position index
	data = []
	for letter in letters:
		mut_seq = seq[:index] + letter + seq[index+1:]
		data.append(('%d%s'%(index,letter),mut_seq))
	return data

def savefig(fig,output_prefix,identifier,ESM_model_name):
	for ext in ['pdf','png']:
		fn = '%s_%s_%s.%s'%(output_prefix,identifier,ESM_model_name,ext)
		fig.savefig(fn)

def step0_prepare(seq, ESM_model_name, model_directory, device):
	#### Load the model
	model, alphabet = just_the_model(ESM_model_name,model_directory)
	model.eval()	# disables dropout for deterministic results
	model = model.to(device) # put onto the gpu
	print('ESM: %s'%(ESM_model_name))
	print('Using Device:',device)
	print('----------')

	#### Setup 
	seq = clean_sequence(seq) ## clean up the input -- only uppercase AAs
	print('Sequence: %s'%(seq))
	print('Length: %d'%(len(seq)))

	indices = get_target_indices(seq,'C') ## check cysteines
	ncys = len(indices)
	print('Num. Cys: %d'%(ncys))
	if ncys == 0:
		print('No Cys to remove. Finished!')
		sys.exit(0)
	print('Cys locations:',*[index+1 for index in indices])
	print('----------')

	return seq,model,alphabet,indices

def step1_calibratemaskless(seqs,model,alphabet,device):
	#### fits PLL from wt marginal data. Seqs is a list [seq1,seq2,seq2]
	## Masks out each position in a sequence: [(0, mask 0), (1, mask 1),..., (len(seq), wt)] one at a time
	## Gets all of the embeddings and then does the fitting....

	print('Learning PLL map...')

	lnp_wt_all = []
	lnp_mask_all = []
	for index in range(len(seqs)):
		seq = clean_sequence(seqs[index]) ## clean up input
		
		### Create masked out sequences (each sequence has one position masked out)
		data = []
		for index in range(len(seq)):
			masked_seq = seq[:index] + '<mask>' + seq[index+1:]
			data += [('%s%d'%(seq[index],index+1),masked_seq)] ## each position in data corresponds to the masked out index
		data.append(('wt',seq)) ## Adds on one more, i.e., the last position is the un-masked WT seq
		
		#### Embed one masked position sequence at a time to avoid computation issues for long sequences/big models
		logits = []
		for i in range(len(data)): ## do it this way b/c of memory constraints for long sequences....
			_data = [data[i],]*3 ## batching issues again! Must do >=3x to ensure every platform gives the same results
			_reps,_logits =  embed_sequences(_data,model,alphabet,device)
			logits.append(_logits[0]) ## just append one of them...
		logits = np.array(logits)

		#### Calculate PLL_i components
		lnp_wt = np.zeros(len(seq))
		lnp_mask = np.zeros(len(seq))
		for i in range(len(seq)):
			## +1 is to skip the cls position ([:,0,:])
			## eos position ([:,-1,:]) is skipped bc loop only goes to len(seq)
			pmaski = softmax(logits[i,i+1])[letterids] ## only get AA positions
			pwti = softmax(logits[-1,i+1])[letterids] ## only get AA positions
			pmaski /= pmaski.sum() ## renormalize
			pwti /= pwti.sum() ## renormalize

			## decode the sequence
			target = letters.index(seq[i])
			lnp_wt[i] = np.log(pwti[target])
			lnp_mask[i] = np.log(pmaski[target])

		## collect all in the fo
		lnp_wt_all.append(lnp_wt)
		lnp_mask_all.append(lnp_mask)

	#### fit the mapping function
	theta,fig,ax = _optimize_maskless_scoring(np.concatenate(lnp_wt_all),np.concatenate(lnp_mask_all))

	#### Output information
	print('----------')
	print('Check Fitting: (Index, Estimate, PPPL)')
	for index in range(len(seqs)):
		print(index, ## which sequence 
			np.exp(-np.mean(pll_fxn(lnp_wt_all[index],*theta))), ##Fitted data
			np.exp(-np.mean(lnp_mask_all[index])) ## actual value
		)
	print('----------')
	return theta,fig,ax

def step2_removecys(seq,model,alphabet,device,theta,history,sequence_order,flag_mask=True):
	wt_seq = clean_sequence(seq)

	#### Mask out all cysteines. Controlled by flag_mask...
	indices = get_target_indices(wt_seq,'C')
	if flag_mask: ### Mask them
		masked_seq = wt_seq.replace('C','<mask>')
	else: ### Don't mask them!!! use WT embedding
		masked_seq = ''.join(list(wt_seq))
	data = [('Masked WT',masked_seq),]*3  ## n.b., running 3x b/c of batching issues with a single sequence

	#### Prepare sequence: convert AAs to tokens
	batch_converter = alphabet.get_batch_converter()
	batch_labels, batch_strs, batch_tokens = batch_converter(data)
	batch_tokens = batch_tokens.to(device)

	#### Embed mMasked WT sequence
	with torch.no_grad():
		results = model(batch_tokens, repr_layers=[model.num_layers,], return_contacts=False)
		logits = results['logits'].cpu().numpy().copy()[0]
	
	#### Figure out mutations at each site
	mutations = []
	for index in indices:
		probs = softmax(logits[index+1]) ## Turn logits into probabilities
		probs = probs[letterids] ## just keep the AAs 
		probs[letters.index('C')] = 0. ## zero out everything we won't consider, i.e., Cys
		probs /= probs.sum() ## renormalize probabilities after conditional of "not Cys"
		mutations.append([wt_seq[index],index,letters[probs.argmax()]]) ## Make choice of best non-Cys residue

	#### Render mutant 
	mut_seq = ''.join(list(wt_seq)) ## make a deep copy
	for mutation in mutations:
		orig,ind,repl = mutation
		mut_seq = mut_seq[:ind] + repl + mut_seq[ind+1:]
	
	#### Render naive 
	ala_seq = wt_seq.replace('C','A')

	## Estimate PPPLs
	data = [('WT  ',wt_seq),('Ala ',ala_seq),('Mut ',mut_seq)]
	reps,logits = embed_sequences(data,model,alphabet,device)
	pppls = logits_to_pppl_approx(data,logits,alphabet,theta)

	## Output Information
	print('2. Initial Replacement')
	print('----------')
	for i in range(len(data)):
		print(data[i][0],":",data[i][1])
	print('2. Initial Mutations:')
	for mutation in mutations: 
		print('\tC%d%s'%(mutation[1]+1,mutation[2]))
	print('----------')
	print('Initial Pseudoperplexities:')
	for i in range(len(data)):
		history[data[i][1]] = [pppls[i],reps[i,0].copy()]
		print(data[i][0],":",history[data[i][1]][0])

	
	sequence_order.append(wt_seq) ## For tracking walk through latent space
	sequence_order.append(mut_seq)
	print('----------\n')
	return wt_seq,ala_seq,mut_seq,mutations,history,sequence_order

def step3_optimize(seq,indices,model,alphabet,device,theta,history,sequence_order,n_rounds):
	mut_seq = ''.join(list(seq)) ## make a deep copy

	print('Step 3: Polishing')
	print('----------')
	for iter in range(n_rounds):
		## get starting point
		data = [('mut',mut_seq),]*3
		reps,logits = embed_sequences(data,model,alphabet,device)
		mut_pp = logits_to_pppl_approx(data,logits,alphabet,theta)[0]
		best = [-1,mut_pp,-1]
		# theta,fig,ax = optimize_maskless_scoring_batch([mut_seq,],model,alphabet,device)

		t0 = time.time()
		for index in indices:
			data = generate_pointmutants(mut_seq,index)
			data = data[:-1] # no C
			reps,logits = embed_sequences(data,model,alphabet,device)
			pp = logits_to_pppl_approx(data,logits,alphabet,theta)
			if pp.min() < best[1]:
				if letters[pp.argmin()] != mut_seq[index]: ## sometimes the comparison fails
					best = [index,pp.min(),pp.argmin()]
			for i in range(len(data)):
				if not data[i][1] in history:
					history[data[i][1]] = [pp[i],reps[i,0].copy()]
		t1 = time.time()

		if 	best[0] != -1:
			print('3.%d Polish: C%d%s, %.3f sec\nMut: %s\nPseudoperplexity: %.8f'%(iter+1,best[0]+1,letters[best[2]],t1-t0,mut_seq,best[1]))
			mut_seq = mut_seq[:best[0]] + letters[best[2]] + mut_seq[best[0]+1:]
			sequence_order.append(mut_seq)
			print('----------')
		else: ## No changes were made
			print('3.%d Polish; %.3f sec, <no better change>'%(iter+1,t1-t0))
			break
	
	## Output Information
	print('\n---------- Final ----------')
	print('MUT Sequence: %s'%(mut_seq))
	data = [('mut',mut_seq),]*3
	reps,logits = embed_sequences(data,model,alphabet,device)
	mut_pp= logits_to_pppl_approx(data,logits,alphabet,theta)[0]
	print('Pseudoperplexity: %.8f'%(mut_pp))
	print('Mutations:')
	for index in indices:
		print('\t%s%d%s'%('C',index+1,mut_seq[index]))

	return mut_seq,history,sequence_order

def plot_pca(output_prefix,ESM_model_name,history,sequence_order,wt_seq,mut_seq,ala_seq):
	q = np.array([history[k][1] for k in history.keys()])
	
	pca = PCA(n_components=2)
	w = pca.fit_transform(q)

	fig,ax = plt.subplots(1,figsize=(8,6))

	keep = np.array([not key in [wt_seq,mut_seq,ala_seq] for key in history.keys()])
	ax.plot(w[keep,0],w[keep,1],'o',color='gray',label='Mutants',alpha=.5)

	keep = np.array([key in [wt_seq,] for key in history.keys()])
	ax.plot(w[keep,0],w[keep,1],'o',color='tab:blue',label='WT')

	keep = np.array([key in [ala_seq,] for key in history.keys()])
	ax.plot(w[keep,0],w[keep,1],'o',color='tab:orange',label='Alanine Control')

	keep = np.array([key in [mut_seq,] for key in history.keys()])
	ax.plot(w[keep,0],w[keep,1],'o',color='tab:red',label='Optimized')

	ww = []
	keylist = [key for key in history.keys()]
	keylist = list(history.keys())
	for soi in sequence_order:
		for ind in range(len(keylist)):
			if keylist[ind] == soi:
				ww.append(w[ind])
				break
	ww = np.array(ww)
	ax.plot(ww[:,0],ww[:,1],color='k',label='Opt. Path',zorder=2,alpha=.8)

	ax.set_xlabel('PCA1')
	ax.set_ylabel('PCA2')
	ax.legend()

	ax.set_title('%s (%s)'%(output_prefix,ESM_model_name))

	fig.tight_layout()
	savefig(fig,output_prefix,'PCA',ESM_model_name)
	plt.show()

def plot_pppl(output_prefix,ESM_model_name,sequence_order,history,wt_seq,mut_seq,ala_seq):
	pps = np.array([history[k][0] for k in history.keys()])
	
	ident = ['',]*(pps.size)
	keylist = list(history.keys())

	for i in range(len(keylist)):
		if keylist[i] == wt_seq:
			ident[i] = 'WT'
			ppw = pps[i]
		elif keylist[i] == mut_seq:
			ident[i] = 'Optimized'
			ppm = pps[i]
		elif keylist[i] == ala_seq:
			ident[i] = 'Alanine Control'
			ppa = pps[i]
		else:
			ident[i] = 'Mutants'

	ident=np.array(ident)

	kde = gaussian_kde(pps)
	xmin = pps.min()
	xmax = pps.max()
	xdelta = xmax-xmin
	factor = .25
	x = np.linspace(xmin - xdelta*factor,xmax+xdelta*factor,1000)
	ykde = kde(x)
	pkde = kde(pps)
	yrvs = np.random.rand(pps.size)*pkde

	fig,ax=plt.subplots(1,figsize=(8,6))

	for target,color in zip(['Mutants','WT','Alanine Control','Optimized'],['gray','tab:blue','tab:orange','tab:red']):
		keep = [identi in [target] for identi in ident]
		ax.plot(pps[keep],yrvs[keep],'o',color=color,alpha=.8,label=target)

	ax.plot(x,ykde,color='k',lw=1.5,)
	ax.set_xlim(x.min(),x.max())
	ax.set_ylabel('Density')
	ax.set_xlabel('Pseudoperplexity')

	ax.axvline(x=ppw,color='tab:blue',zorder=-5,lw=2)
	ax.axvline(x=ppa,color='tab:orange',zorder=-5,lw=2)
	ax.axvline(x=ppm,color='tab:red',zorder=-5,lw=2)
	ax.legend()

	ax.set_title('%s (%s)'%(output_prefix,ESM_model_name))

	fig.tight_layout()
	savefig(fig,output_prefix,'PPPL',ESM_model_name)
	plt.show()

def main(input_sequence,ESM_model_name="esm2_t6_8M_UR50D",device='cpu',n_rounds=20,no_plots=False,output_prefix='test',model_directory=None):
	print(header)

	if len(clean_sequence(input_sequence)) < 1:
		raise Exception('Enter a valid sequence')

	######## INITIALIZE
	#### Step 0: Prepare things
	seq,model,alphabet,indices = step0_prepare(input_sequence,ESM_model_name,model_directory,device)

	##### Step 1: Map PLL for maskless scoring
	theta,fig,ax = step1_calibratemaskless([seq,],model,alphabet,device)
	if not no_plots:
		savefig(fig,output_prefix,'map',ESM_model_name)
		plt.show()
	
	######### OPTIMIZATION
	print('\n---------- Optimization ----------')
	history = {}
	sequence_order = []

	#### Step 2: Initial Replacement
	wt_seq,ala_seq,mut_seq,mutations,history,sequence_order = step2_removecys(seq,model,alphabet,device,theta,history,sequence_order)

	#### Step 3: Optimize Sequence
	mut_seq, history, sequence_order = step3_optimize(mut_seq,indices,model,alphabet,device,theta,history,sequence_order,n_rounds)

	#### Step 4: Plots
	if not no_plots:
		## Latent space
		plot_pca(output_prefix,ESM_model_name,history,sequence_order,wt_seq,mut_seq,ala_seq)
		## Score histogram
		plot_pppl(output_prefix,ESM_model_name,sequence_order,history,wt_seq,mut_seq,ala_seq)

def cli():
	parser = argparse.ArgumentParser(
		description=header,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)

	parser.add_argument("sequence", type=str, help="WT protein sequence to alter")
	parser.add_argument("--model", choices=ESM_models, default=ESM_models[0], help='Which ESM2 model to use?')
	parser.add_argument("--device", choices=devices, default=devices[0], help="Which compute device?")

	parser.add_argument("--model_directory", default = None, help="Where to save/load the ESM model files")
	
	parser.add_argument("--n_rounds", type=int, default=20, help="Maximum Number of Polishing Rounds")

	parser.add_argument("--output_prefix", type=str, default='', help="Choose a prefix to save the images")
	parser.add_argument("--no_plots",action="store_true", help="Do not display any plots")

	args = parser.parse_args()
	main(args.sequence,args.model,args.device,args.n_rounds,args.no_plots,args.output_prefix,args.model_directory)

if __name__ == "__main__":
	cli()