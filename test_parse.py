from analysis.transcript_parsing import parse
import analysis.preprocess as prep
import portion as P
seg = list(prep.silence_index['Bmr024']['fe008'])
#print(list(seg[0]|P.closed(537,538)))
p = seg[-1]
print(p.lower)
print(p.upper)
du = p.upper - p.lower
new_seg = P.open(p.lower, p.lower+du)
print(prep.silence_index['Bmr024']['fe008'].contains(new_seg))
