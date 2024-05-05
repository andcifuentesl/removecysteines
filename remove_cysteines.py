import torch
import esm
import numpy as np
from scipy.special import softmax
import time
import sys
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="esm")

letters = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C']
letterids = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])

def _embed_sequences(data,model,batch_converter,device):
	batch_labels, batch_strs, batch_tokens = batch_converter(data)

	batch_tokens = batch_tokens.to(device)
	with torch.no_grad():
		results = model(batch_tokens, repr_layers=[model.num_layers,], return_contacts=False)

	reps = results['representations'][model.num_layers].cpu().numpy().copy()
	logits = results['logits'].cpu().numpy().copy()

	return reps,logits

def embed_sequences(data,model,batch_converter,device):
	## FIX: this is busted for N = 2....
	if len(data) < 3: ## ensure everything is always a batch... of three
		reps,logits = _embed_sequences(data*3,model,batch_converter,device)
		return reps[0][None,:,:],logits[0][None,:,:]
	return _embed_sequences(data,model,batch_converter,device)

def generate_pointmutants(sequence,index):
	data = []
	for letter in letters:
		mutated_sequence = sequence[:index] + letter + sequence[index+1:]
		data.append(('%d%s'%(index,letter),mutated_sequence))
	return data

def calc_pseudoperplexity(logits,seq):
	#### eqn 4
	#### logits (seq,latent)
	#### sequence (seq)

	#### calculate probabilities
	probs = np.exp(logits)
	probs /= np.sum(probs,axis=1)[:,None]

	## decode sequence
	seq_ids = np.array([letterids[letters.index(seq[i])] for i in range(len(seq))])
	nlp = -np.log(probs[1:-1,seq_ids]) ## remove CLS and EOS tokens.

	## calculate pseudoperplexity
	pppl = np.exp(np.mean(nlp))
	return pppl

def calc_pseudoperplexities(logits,data):
	perp = np.array([calc_pseudoperplexity(logits[i],data[i][1]) for i in range(len(data))])
	return perp

def calc_given_best(sequence,model,batch_converter,indices,device):
	data = [('wt',sequence)]
	reps, logits = embed_sequences(data,model,batch_converter,device)

	out = []
	for index in indices:
		p = softmax(logits[0,index+1,letterids])
		p[letters.index(sequence[index])] = 0.
		p /= p.sum()
		out.append([sequence[index],index,letters[p.argmax()]])
	return out

def just_the_model(model_name):
	url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
	model_data = esm.pretrained.load_hub_workaround(url)
	return esm.pretrained.load_model_and_alphabet_core(model_name, model_data, None)

def diff_str(s1,s2):
	'''1>2'''
	diffs = []
	for i in range(len(s1)):
		if s1[i] != s2[i]:
			diffs.append('%s%d%s'%(s1[i],i+1,s2[i]))
	if len(diffs) == 0:
		return 'WT'
	return ' '.join(diffs)


def main(wt_sequence, ESM_model_name='esm2_t33_650M_UR50D', device='cpu', n_rounds=20, show_pca=True,show_pp=True):

	#### Load ESM-2 model
	model, alphabet = just_the_model(ESM_model_name)
	print('ESM: %s'%(ESM_model_name))

	#### Find GPU type
	print('Using Device:',device)
	print('----------')

	batch_converter = alphabet.get_batch_converter()
	model.eval()	# disables dropout for deterministic results
	model = model.to(device) # put onto the gpu

	#### Clean input sequence
	wt_sequence = ''.join([si for si in wt_sequence if si in letters])

	#### Reporting Statistics
	print('WT Sequence: %s'%(wt_sequence))
	## start record keeping
	cls = {}
	lnls = {}
	sequence_order = [wt_sequence]

	reps,logits = embed_sequences([('wt',wt_sequence),]*3,model,batch_converter,device)
	wt_pp = calc_pseudoperplexity(logits[0],wt_sequence)
	print('WT Perplexity: %.16f'%(wt_pp))
	cls[wt_sequence] = reps[0,0]
	lnls[wt_sequence] = wt_pp

	print('Length: %d'%(len(wt_sequence)))

	ncys = wt_sequence.count('C')
	print('Num. Cys: %d'%(ncys))
	if ncys == 0:
		print('No Cys to remove. Finished!')
		sys.exit(0)

	print('Cys locations:',*[index+1 for index in range(len(wt_sequence)) if wt_sequence[index] == 'C'])

	#### AA control 
	ala_sequence = ''.join(list(wt_sequence)) ## make a deep copy
	indices = np.array([index for index in range(len(ala_sequence)) if ala_sequence[index] == 'C'])
	for ind in indices:
		ala_sequence = ala_sequence[:ind] + 'A' + ala_sequence[ind+1:]
	reps,logits = embed_sequences([('ala',ala_sequence),]*3,model,batch_converter,device) ## note you need to run at least 3 for reproducibility ('batch' issues)
	ala_pp = calc_pseudoperplexity(logits[0],ala_sequence)
	cls[ala_sequence] = reps[0,0]
	lnls[ala_sequence] = ala_pp

	##################### Design
	print('\n---------- Optimization ----------')
	mut_sequence = ''.join(list(wt_sequence)) ## make a deep copy
	indices = np.array([index for index in range(len(mut_sequence)) if mut_sequence[index] == 'C'])

	#### Step 1. Remove all C using the best (unmasked) alternative
	mutations = calc_given_best(mut_sequence,model,batch_converter,indices,device)
	for mutation in mutations:
		orig,ind,repl = mutation
		mut_sequence = mut_sequence[:ind] + repl + mut_sequence[ind+1:]

	reps,logits = embed_sequences([('mut',mut_sequence),]*3,model,batch_converter,device) ## note you need to run at least 3 for reproducibility ('batch' issues)
	mut_pp = calc_pseudoperplexity(logits[0],mut_sequence)
	cls[mut_sequence] = reps[0,0]
	lnls[mut_sequence] = mut_pp
	print('1. Initial MUT perplexity: %.8f'%(mut_pp))
	for mutation in mutations:
		print('\tC%d%s'%(mutation[1]+1,mutation[2]))
	sequence_order.append(mut_sequence)

	#### Step 2. Scan all point changes to maximize perplexity
	## optimization rounds
	for iter in range(n_rounds):
		## get starting point
		reps,logits = embed_sequences([('mut',mut_sequence),]*3,model,batch_converter,device)
		mut_pp = calc_pseudoperplexity(logits[0],mut_sequence)
		best = [-1,mut_pp,-1]

		t0 = time.time()
		for index in indices:
			data = generate_pointmutants(mut_sequence,index)
			data = data[:-1] # no C
			reps,logits = embed_sequences(data,model,batch_converter,device)
			pp = calc_pseudoperplexities(logits,data)
			if pp.max() > best[1]:
				if letters[pp.argmax()] != mut_sequence[index]: ## sometimes the comparison fails
					best = [index,pp.max(),pp.argmax()]
			for i in range(len(data)):
				if not data[i][1] in cls:
					cls[data[i][1]] = reps[i,0].copy()
					lnls[data[i][1]] = pp[i]
		t1 = time.time()

		if 	best[0] != -1:
			print('2.%d Polish, MUT perplexity %.8f, %.3f sec, C%d%s'%(iter+1,best[1],t1-t0,best[0]+1,letters[best[2]]))
			mut_sequence = mut_sequence[:best[0]] + letters[best[2]] + mut_sequence[best[0]+1:]
			sequence_order.append(mut_sequence)
		else:
			print('2.%d Polish, MUT perplexity %.8f, %.3f sec, <no better change>'%(iter+1,best[1],t1-t0))
			break

	#### Step 3. Finish up
	print('\n---------- Final ----------')
	print('MUT Sequence: %s'%(mut_sequence))
	reps,logits = embed_sequences([('mut',mut_sequence),]*3,model,batch_converter,device)
	mut_pp = calc_pseudoperplexity(logits[0],mut_sequence)
	print('MUT Perplexity: %.8f'%(mut_pp))
	print('Mutations:')
	for index in indices:
		print('\t%s%d%s'%(wt_sequence[index],index+1,mut_sequence[index]))

	#### output debug data
	if False:
		out = ''
		out += 'wt ' + ', ,' + wt_sequence + '\n'
		out += 'mut' + ', ,' + mut_sequence + '\n'
		for key in cls.keys():
			line = str(diff_str(wt_sequence,key)) + ',' + str(lnls[key]) + ',' + str(key)
			out += line + '\n'
		with open('temp.csv','w') as f:
			f.write(out)


	#### Step 4. Analysis
	if show_pca:
		q = np.array([cls[k] for k in cls.keys()])
		
		import matplotlib.pyplot as plt
		from sklearn.decomposition import PCA
		pca = PCA(n_components=2)
		w = pca.fit_transform(q)

		keep = np.array([not key in [wt_sequence,mut_sequence,ala_sequence] for key in cls.keys()])
		plt.plot(w[keep,0],w[keep,1],'o',color='gray',label='Mutants')

		keep = np.array([key in [wt_sequence,] for key in cls.keys()])
		plt.plot(w[keep,0],w[keep,1],'o',color='tab:blue',label='WT')

		keep = np.array([key in [ala_sequence,] for key in cls.keys()])
		plt.plot(w[keep,0],w[keep,1],'o',color='tab:orange',label='Alanine Control')

		keep = np.array([key in [mut_sequence,] for key in cls.keys()])
		plt.plot(w[keep,0],w[keep,1],'o',color='tab:red',label='Optimized')

		ww = []
		keylist = [key for key in cls.keys()]
		keylist = list(cls.keys())
		for soi in sequence_order:
			for ind in range(len(keylist)):
				if keylist[ind] == soi:
					ww.append(w[ind])
					break
		ww = np.array(ww)
		plt.plot(ww[:,0],ww[:,1],color='k',label='Opt. Path',zorder=2,alpha=.8)

		plt.xlabel('PCA1')
		plt.ylabel('PCA2')
		plt.legend()
		plt.show()

	if show_pp:
		pps = np.array([lnls[k] for k in lnls.keys()])
		import matplotlib.pyplot as plt
		from scipy.stats import gaussian_kde
		
		ident = ['',]*(pps.size)
		keylist = list(lnls.keys())

		for i in range(len(keylist)):
			if keylist[i] == wt_sequence:
				ident[i] = 'WT'
				ppw = pps[i]
			elif keylist[i] == mut_sequence:
				ident[i] = 'Optimized'
				ppm = pps[i]
			elif keylist[i] == ala_sequence:
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

		fig,ax=plt.subplots(1)

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
	
		plt.show()





if __name__ == "__main__":
	ESM_models = ["esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"]
	
	devices = []
	if torch.cuda.is_available():
		devices.append('cuda')
	if torch.backends.mps.is_available():
		devices.append('mps')
	devices.append('cpu')

	parser = argparse.ArgumentParser(description=r"Remove Cysteines (v0.1.4). CKT Lab -- http://ckinzthompson.github.io")
	parser.add_argument("sequence", type=str, help="WT protein sequence to alter")
	parser.add_argument("--n_rounds", type=int, default=20, help="Maximum Number of Polishing Rounds")
	parser.add_argument("--model", choices=ESM_models, default=ESM_models[3], help='Which ESM2 model to use?')
	parser.add_argument("--device", choices=devices, default=devices[0], help="Which compute device?")
	parser.add_argument("--pca", action="store_true", help="Show embedding PCA?")
	parser.add_argument("--pp", action="store_true", help="Show pseudoperlexities?")
		
	args = parser.parse_args()
	main(args.sequence,args.model,args.device,args.n_rounds,args.pca,args.pp)