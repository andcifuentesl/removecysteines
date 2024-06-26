# Notes 

* Note: on MPS it's essential to iterate rather than batch, b/c of memory pressure issues. I noticed completely wrong values popping up b/c of swapping (I think). External validation here: https://huggingface.co/docs/diffusers/en/optimization/mps
* CPU/GPU batch/iterate give slightly different results (computational error ~1e-6 abs difference). This is probably related to matrix multiplication issues (i.e., different libraries for cpu vs gpu and different functions for 1D vs ND).
* apparently the batch effect comes for N = 3, i.e.,
```
f([A,]) = [a,]
f([A,A,]) = [a,a,]
f([A,A,A,]) = [b,b,b,]
```

## To Do...
...

## Test Sequences

### MS2
>MS2
ASNFTQFVLVDNGGTGDVTVAPSNFANGVAEWISSNSRSQAYKVTCSVRQSSAQKRKYTIKVEVPKVATQTVGGVELPVAAWRSYLNMELTIPIFATNSDCELIVKAMQGLLKDGNPIPSAIAANSGLY

### RecQ
>recQ
MAQAEVLNLESGAKQVLQETFGYQQFRPGQEEIIDTVLSGRDCLVVMPTGGGKSLCYQIPALLLNGLTVVVSPLISLMKDQVDQLQANGVAAACLNSTQTREQQLEVMTGCRTGQIRLLYIAPERLMLDNFLEHLAHWNPVLLAVDEAHCISQWGHDFRPEYAALGQLRQRFPTLPFMALTATADDTTRQDIVRLLGLNDPLIQISSFDRPNIRYMLMEKFKPLDQLMRYVQEQRGKSGIIYCNSRAKVEDTAARLQSKGISAAAYHAGLENNVRADVQEKFQRDDLQIVVATVAFGMGINKPNVRFVVHFDIPRNIESYYQETGRAGRDGLPAEAMLFYDPADMAWLRRCLEEKPQGQLQDIERHKLNAMGAFAEAQTCRRLVLLNYFGEGRQEPCGNCDICLDPPKQYDGSTDAQIALSTIGRVNQRFGMGYVVEVIRGANNQRIRDYGHDKLKVYGMGRDKSHEHWVSVIRQLIHLGLVTQNIAQHSALQLTEAARPVLRGESSLQLAVPRIVALKPKAMQKSFGGNYDRKLFAKLRKLRKSIADESNVPPYVVFNDATLIEMAEQMPITASEMLSVNGVGMRKLERFGKPFMALIRAHVDGDDEE

### mRuby3
> mruby3
MSKGEELIKENMRMKVVMEGSVNGHQFKCTGEGEGRPYEGVQVMRIKVIEGGPLPFAFDILATSFMYGSRTFIKYPADIPDFFKQSFPEGFTWERVTRYEDGGVVTVTQDTSLEDGELVYNVKVRGVNFPSNGPVMQKKTKGWEPNTEMMYPADGGLRGYTDIALKVDGGGHLHCNFVTTYRSKKTVGNIKMPGVHAVDHRLERIEESDNETYVVQREVAVAKYSNLGGGMDELYK

### T4 Lysozyme L99A
> T4 Lysozyme L99A
MNIFEMLRIDERLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRCAAINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSIWYNQTPNRAKRVITTFRTGTWDAYKNL

### GFP
> GFP
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK
