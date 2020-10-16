import collections
import sys

class PhraseProperties :
	def __init__(self, Phrase, Frequency, RFC, PMI, CMI, PKL, IDF, NPQ) :
		self.Phrase = Phrase
		self.Frequency = Frequency
		self.RFC = RFC	# Relative FrequenCy
		self.PMI = PMI	# maximum Pointwise Mutual Information (subset)
		self.CMI = CMI	# maximum Connecting Mutual Information (supset)
		self.PKL = PKL	# Pointwise Kullback-Leibler divergence
		self.IDF = IDF	# Inverse Document Frequency
		self.NPQ = NPQ	# Normalized Phrase Quality (to be calculated)
		return

def LengthPenalty(Constant, Length) :
	return pow(Length, Constant) - 1

# Cut the whole corpus with respect to Normalized Phrase Quality (NPQ)
def CutCorpus(Corpus, LengthPenaltyConstant, NPQ) :
	LenCorpus = len(Corpus)
	Dp = [0.0]
	Cut = [0]
	for i in range(1, LenCorpus+1) :
		Dp.append(-1000000000.0)
		Cut.append(0)
		Dp[i] = Dp[i-1]
		Cut[i] = i-1
		Cur = ''
		for j in range(1, min(i+1, 20)) :
			Cur = Corpus[i-j] + Cur
			if Cur not in NPQ :
				continue
			NewDP = Dp[i-j] + NPQ[Cur] + LengthPenalty(LengthPenaltyConstant, j)
			if NewDP > Dp[i] :
				Dp[i] = NewDP
				Cut[i] = i-j
	return Cut

# Recalculate Normalized Phrase Quality after dividing the whole corpus into phrases
def ReCalculateNPQ(Corpus, Cut, LengthPenaltyConstant, PhraseFrequency, OldNPQ) :
	LenCorpus = len(Corpus)
	LenNPQ = len(OldNPQ)
	NewNPQ = OldNPQ
	Occurence = collections.defaultdict(int)
	Pos = LenCorpus
	while Pos :
		NextPos = Cut[Pos]
		Phrase = Corpus[NextPos:Pos]
		if Phrase in PhraseFrequency :
			Occurence[Phrase] = Occurence[Phrase] + 1
		Pos = NextPos
	for Phrase in NewNPQ :
		NewNPQ[Phrase] = NewNPQ[Phrase] + (Occurence[Phrase]+1.0)/PhraseFrequency[Phrase]
	Sum = 0.0
	for Phrase in NewNPQ :
		Sum = Sum + NewNPQ[Phrase]
	for Phrase in NewNPQ :
		NewNPQ[Phrase] = NewNPQ[Phrase] * LenNPQ / Sum
	return NewNPQ

# Get phrases that appears more than $Threshold <int> in $Corpus <string>, and $DivideSet <set> means all the punctuation marks
def FrequentPhraseDetection(Corpus, Threshold, DivideSet) :
	sys.stderr.write('In FrequentPhraseDetection() : Detecting frequent phrases.\n')
	Len = len(Corpus)
	Frequency = collections.defaultdict(int)
	Index  = collections.defaultdict(set)
	for i in range(Len) :
		if Corpus[i] not in DivideSet :
			Index[Corpus[i]].add(i)
	while len(Index) :
		sys.stderr.write('In FrequentPhraseDetection() : Remaining endpos set size ' + str(len(Index)) + '.\n')
		Index_2 = collections.defaultdict(set)
		for CurPhrase in Index :
			if (len(Index[CurPhrase]) >= Threshold) :
				Frequency[CurPhrase] = len(Index[CurPhrase])
				for Pos in Index[CurPhrase] :
					if (Pos+1 < Len) and (Corpus[Pos+1] not in DivideSet) :
						NxtPhrase = CurPhrase + Corpus[Pos+1]
						Index_2[NxtPhrase].add(Pos+1)
		Index = Index_2
	return Frequency