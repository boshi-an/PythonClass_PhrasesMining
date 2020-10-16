import collections

from modules import Phrase as P

def TextRank(PhraseDict, DivideSet, Doc, K, IterationRound, D, PenaltyConstant) :
	PhraseNPQ = collections.defaultdict(float)
	for Phrase in PhraseDict :
		PhraseNPQ[Phrase] = PhraseDict[Phrase].NPQ
	Cut = P.CutCorpus(Doc, PenaltyConstant, PhraseNPQ)
	PhraseList = []
	PhraseID = collections.defaultdict(int)
	Pos = len(Doc)
	while Pos :
		Pre = Cut[Pos]
		Phrase = Doc[Pre:Pos]
		if Phrase in PhraseDict : PhraseList.append(Phrase)
		elif Phrase in DivideSet : PhraseList.append('#')
		else : PhraseList.append('.')
		Pos = Pre
	PhraseList.reverse()
	N = 1
	for Phrase in PhraseList :
		if Phrase == '#' : continue 
		if Phrase == '.' : continue
		if not PhraseID[Phrase] :
			PhraseID[Phrase] = N
			N += 1
	LenPhraseList = len(PhraseList)
	Edge = [[] for i in range(0, N+1)]
	Rank = [1-D for i in range(0, N+1)]
	for i in range(0, LenPhraseList) :
		if PhraseList[i] == '#' : continue
		for j in range(1, min(K, LenPhraseList-i)) :
			if PhraseList[i+j] == '#' : break
			a = PhraseID[PhraseList[i]]
			b = PhraseID[PhraseList[i+j]]
			if PhraseList[i]=='.' or PhraseList[i+j]=='.' : continue
			Edge[a].append(b)
			Edge[b].append(a)
	for i in range(0, IterationRound) :
		NRank = [1-D for j in range(0, N+1)]
		for x in range(1, N+1) :
			NRank[x] = 1-D
			Out = 0
			for y in Edge[x] : Out += Rank[y]
			for y in Edge[x] : NRank[x] += D * Rank[y] / Out
			Rank = NRank
	PhraseRank = collections.defaultdict(float)
	for Phrase in PhraseID :
		PhraseRank[Phrase] = Rank[PhraseID[Phrase]]
	return PhraseRank