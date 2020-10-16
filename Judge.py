import collections
from modules import Phrase as P

def GetDict(F) :
	Dict = set()
	while True :
		Line = F.readline()
		if not Line :
			break
		Splited = Line.split()
		Dict.add(Splited[0])
	return Dict

def SetVal(S) :
	Val = 0.0
	for s in S :
		Val += P.LengthPenalty(1.5, len(s))
	return Val

def Similarity(S1, S2) :
	Intersection = S1.intersection(S2)
	Union = S1.union(S2)
	return SetVal(Intersection) / SetVal(Union)

def main() :
	XinDictFile1 = open('./result_xin_1/Dic.txt')
	XinDictFile2 = open('./result_xin_2/Dic.txt')
	ZbnDictFile1 = open('./result_zbn_1/Dic.txt')
	ZbnDictFile2 = open('./result_zbn_2/Dic.txt')
	XinDict1 = GetDict(XinDictFile1)
	XinDict2 = GetDict(XinDictFile2)
	ZbnDict1 = GetDict(ZbnDictFile1)
	ZbnDict2 = GetDict(ZbnDictFile2)
	print('Similarity between Xin1 & Zbn1 is ', Similarity(XinDict1, ZbnDict1))
	print('Similarity between Xin2 & Zbn2 is ', Similarity(XinDict2, ZbnDict2))
	print('Similarity between Xin1 & Xin2 is ', Similarity(XinDict1, XinDict2))
	print('Similarity between Zbn1 & Zbn2 is ', Similarity(ZbnDict1, ZbnDict2))

main()