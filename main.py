# -*- Coding:UTF-8 -*-
import math
import re
import collections
import sys
from queue import Queue
from sklearn.ensemble import RandomForestClassifier
import random

from modules import ACAutomaton as A
from modules import Phrase as P
from modules import TextRank as T

# Read articles of all days from a file located in $FilePath <string>, return a <dict>
def ReadDayArticle(FilePath) :
	sys.stderr.write('In ReadDayArticle() : Reading File ' + FilePath + '.\n')
	DayCorpus = collections.defaultdict(str)
	FileOperator = open(FilePath)
	Content = FileOperator.read()
	Content = Content.replace('\n', '')
	Pattern = re.compile(r'<DOC id=[\s\S]*?</DOC>')
	Result = Pattern.findall(Content)
	for Article in Result :
		Corpus = ''
		Date = Article[17:25]
		PatternHeadline = re.compile(r'(?<=<HEADLINE>)[\s\S]*?(?=</HEADLINE>)')
		Headline = PatternHeadline.findall(Article)
		for Line in Headline :
			Corpus = Corpus + Line
		PatternDateline = re.compile(r'(?<=<DATELINE>)[\s\S]*?(?=</DATELINE>)')
		Dateline = PatternDateline.findall(Article)
		for Line in Dateline :
			Corpus = Corpus + Line
		PatternParagraph = re.compile(r'(?<=<P>)[\s\S]*?(?=</P>)')
		Paragraph = PatternParagraph.findall(Article)
		for Line in Paragraph :
			Corpus = Corpus + Line
		DayCorpus[Date] = DayCorpus[Date] + Corpus + '#'
	return DayCorpus

# Calculate Inverse Document Frequency
def CalcIDF(DocDict, PhraseFrequency) :
	sys.stderr.write('In CalcIDF() : Calculating IDF .\n')
	AC = A.AhoCorasickAutomaton(PhraseFrequency)
	Occurence = collections.defaultdict(int)
	LenDocDict = len(DocDict)
	for Doc in DocDict :
		Occured = AC.Match(DocDict[Doc])
		for Phrase in Occured :
			Occurence[Phrase] += 1
	IDF = collections.defaultdict(float)
	for Phrase in Occurence :
		IDF[Phrase] = math.log(1.0 * LenDocDict / (Occurence[Phrase]+1))
	return IDF

def CalcAbstractTFIDF(DocDict, PhraseDict, RequiredNum) :
	sys.stderr.write('In CalcAbstract() : Calculating abstract phrases for each document .\n')
	Abstract = collections.defaultdict(list)
	AC = A.AhoCorasickAutomaton(PhraseDict)
	for Doc in DocDict :
		# sys.stderr.write('In CalcAbstract() : Calculating abstract phrases for ' + Doc + '.\n')
		Occured = AC.Match(DocDict[Doc])
		TF = collections.defaultdict(float)
		TFIDF = collections.defaultdict(float)
		TFIDFQ = collections.defaultdict(float)
		for Phrase in Occured :
			TF[Phrase] = 1.0 * Occured[Phrase] / PhraseDict[Phrase].Frequency
			TFIDF[Phrase] = TF[Phrase] * PhraseDict[Phrase].IDF
			TFIDFQ[Phrase] = TFIDF[Phrase] * PhraseDict[Phrase].NPQ
		PossibleAbstractList = sorted(TFIDFQ.items(), key = lambda d:d[1], reverse = True)
		Abstract[Doc] = PossibleAbstractList[:min(RequiredNum, len(PossibleAbstractList))]
	return Abstract
		
def TrainClassifier(Data, Target) :
	sys.stderr.write('In TrainClassifier() : Training Random Forest Classifier .\n')
	assert(len(Data) == len(Target))
	Len = len(Data)
	DataLen = int(0.75 * Len)
	Classifier = RandomForestClassifier()
	Classifier.fit(Data[DataLen:], Target[DataLen:])
	sys.stderr.write('In TrainClassifier() : Trained score on training data : ' + str(Classifier.score(Data[DataLen:], Target[DataLen:])) + '.\n')
	sys.stderr.write('In TrainClassifier() : Trained score on testing data : ' + str(Classifier.score(Data[:DataLen], Target[:DataLen])) + '.\n')
	return Classifier

def EvaluatePhrase(Classifier, Phrases) :
	Ans = []
	Res = Classifier.predict_proba(Phrases)
	for Tmp in Res :
		Ans.append(Tmp[1])
	return Ans

def DeleteUnexisted(Dict, Model) :
	Copy = Dict.copy()
	for Phrase in Copy :
		if Phrase not in Model :
			Dict.pop(Phrase)

def main() :
	print('Program started!')
	IterationRound = 60
	FrequencyLimit = 10
	PenaltyConstant = 2
	NeededPercentage = 25
	TextRankK = 4
	TextRankD = 0.85
	RequiredTFIDFNum = 25
	RequiredTextRankNum = 4
	MaxThread = 2
	ThreadBlock = 1024
	assert(ThreadBlock % MaxThread == 0)
	DateList = [
		#'xin_cmn_200010',
		#'zbn_cmn_200010',
		#'xin_cmn_200011',
		'zbn_cmn_200011'
		#'xin_cmn_200012',
		#'xin_cmn_200101',
		#'xin_cmn_200304',
		#'xin_cmn_200305',
		#'xin_cmn_200306',
		#'xin_cmn_200307',
		#'xin_cmn_200308',
		#'xin_cmn_200309'
	]
	DivideSet = {',', '・', '。', '“', '”', '(', ')', '?', '!', ':', '　', '―', '─', '、', ';', '-', ' ', '》', '《', '.', '#', '）', '：', '（', '！', '，'}
	Log = open('./result/Log.txt', 'w')
	Dic = open('./result/Dic.txt', 'w')
	Abs = open('./result/Abs.txt', 'w')
	Rnk = open('./result/Rnk.txt', 'w')
	Cor = open('./result/Cor.txt', 'w')
	DataFile = open('./data/TrainingData.txt', 'r')
	Corpus = ''
	DocDict = collections.defaultdict(str)
	PhraseDict = collections.defaultdict(P.PhraseProperties)

	for Date in DateList :
		DayCorpus = ReadDayArticle('./data/' + Date)
		DocDict.update(DayCorpus)
		for Day in DayCorpus :
			Corpus = Corpus + DayCorpus[Day]
	print(Corpus, file = Cor)
	PhraseFrequency = P.FrequentPhraseDetection(Corpus, FrequencyLimit, DivideSet)
	sys.stderr.write('Frequent phrases estimated! ' + str(len(PhraseFrequency)) + ' Phrases.\n')
	
	#for Phrase in PhraseFrequency :
	#	print(Phrase, PhraseFrequency[Phrase], file = Log)

	PhraseRFC = collections.defaultdict(float)
	PhrasePMI = collections.defaultdict(float)
	PhraseCMI = collections.defaultdict(float)
	PhrasePKL = collections.defaultdict(float)
	PhraseIDF = collections.defaultdict(float)
	PhraseNPQ = collections.defaultdict(float)

	Sum = 0
	for Phrase in PhraseFrequency :
		Sum += PhraseFrequency[Phrase]
	for Phrase in PhraseFrequency :
		PhraseRFC[Phrase] = 1.0 * PhraseFrequency[Phrase] / Sum
	
	for Phrase in PhraseFrequency :
		PhrasePMI[Phrase] = +1000
		PhraseCMI[Phrase] = -1000
		PhrasePKL[Phrase] = +1000
		PhraseIDF[Phrase] = -1000

	for Phrase in PhraseFrequency :
		Len = len(Phrase)
		for i in range(1, Len) :
			u = Phrase[:i]
			v = Phrase[i:]
			PhrasePMI[Phrase] = min(PhrasePMI[Phrase], math.log(PhraseRFC[Phrase] / PhraseRFC[u] / PhraseRFC[v]))
			PhraseCMI[u] = max(PhraseCMI[u], math.log(PhraseRFC[Phrase] * PhraseRFC[Phrase] / PhraseRFC[u] / PhraseRFC[v]))
			PhraseCMI[v] = max(PhraseCMI[v], math.log(PhraseRFC[Phrase] * PhraseRFC[Phrase] / PhraseRFC[u] / PhraseRFC[v]))
			PhrasePKL[Phrase] = min(PhrasePKL[Phrase], PhraseRFC[Phrase] * PhrasePMI[Phrase])
	PhraseIDF = CalcIDF(DocDict, PhraseFrequency)
	
	for Phrase in PhraseFrequency :
		PhraseDict[Phrase] = P.PhraseProperties(Phrase,\
											PhraseFrequency[Phrase],\
											PhraseRFC[Phrase],\
											PhrasePMI[Phrase],\
											PhraseCMI[Phrase],\
											PhrasePKL[Phrase],\
											PhraseIDF[Phrase],\
											0)
	
	TrainingData = []
	TrainingTarget = []
	
	while True :
		Line = DataFile.readline()
		if not Line :
			break
		Splited = Line.split()
		TrainingData.append([float(i) for i in Splited[2:]])
		TrainingTarget.append(Splited[0])
	
	Classifier = TrainClassifier(TrainingData, TrainingTarget)

	sys.stderr.write('In main() : Evaluating phrases.\n')
	LenNPQ = len(PhraseFrequency)
	Len = len(PhraseDict)

	DataList = []
	AnsList = []
	for Phrase in PhraseDict :
		DataList.append([PhraseDict[Phrase].RFC,\
						PhraseDict[Phrase].PMI,\
						PhraseDict[Phrase].CMI,\
						PhraseDict[Phrase].PKL,\
						PhraseDict[Phrase].IDF])
	AnsList = EvaluatePhrase(Classifier, DataList)
	Cur = 0
	for Phrase in PhraseDict :
		PhraseNPQ[Phrase] = AnsList[Cur]
		Cur += 1
	'''
	Cur = 0
	DataList = []
	AnsList = []
	for Phrase in PhraseDict :
		DataList.append([PhraseDict[Phrase].RFC,\
						PhraseDict[Phrase].PMI,\
						PhraseDict[Phrase].CMI,\
						PhraseDict[Phrase].PKL,\
						PhraseDict[Phrase].IDF])
	while Cur < Len :
		PerThread = ThreadBlock // MaxThread
		ThreadList = []
		for i in range(0, MaxThread) :
			if Cur+i*PerThread >= Len : break
			EvalutaionThread = MyThread(EvaluatePhrase, args = (Classifier, DataList[Cur+i*PerThread : min(Cur+(i+1)*PerThread, Len)]))
			ThreadList.append(EvalutaionThread)
			EvalutaionThread.start()
		for Thread in ThreadList :
			Thread.join()
			AnsList.extend(Thread.get_result())
		Cur += ThreadBlock
		sys.stderr.write('\tProgress : ' + str(Cur) + '/' + str(Len) + ' .\n')
	Cur = 0
	for Phrase in PhraseDict :
		PhraseNPQ[Phrase] = AnsList[Cur]
		Cur += 1
	'''
	'''
	Cur = 0
	for Phrase in PhraseDict :
		PhraseNPQ[Phrase] = Classifier.predict_proba([[PhraseDict[Phrase].RFC,\
													PhraseDict[Phrase].PMI,\
													PhraseDict[Phrase].CMI,\
													PhraseDict[Phrase].PKL,\
													PhraseDict[Phrase].IDF]])[0][1]
		assert(Classifier.predict_proba([[PhraseDict[Phrase].RFC,\
													PhraseDict[Phrase].PMI,\
													PhraseDict[Phrase].CMI,\
													PhraseDict[Phrase].PKL,\
													PhraseDict[Phrase].IDF]])[0][1] == PhraseNPQ[Phrase])
		Cur += 1
		if Cur%1000 == 0 :
			sys.stderr.write('\tProgress : ' + str(Cur) + '/' + str(Len) + ' .\n')
	'''

	Sum = 0.0
	for Phrase in PhraseDict :
		Sum += PhraseNPQ[Phrase]
	for Phrase in PhraseDict :
		PhraseNPQ[Phrase] = PhraseNPQ[Phrase] * LenNPQ / Sum

	for i in range(0, IterationRound) :
		sys.stderr.write('In main() : iteration Round ' + str(i) + ', Phrase set size : ' + str(len(PhraseNPQ)) + '.\n')
		Cut = P.CutCorpus(Corpus, PenaltyConstant, PhraseNPQ)
		PhraseNPQ = P.ReCalculateNPQ(Corpus, Cut, PenaltyConstant, PhraseFrequency, PhraseNPQ)
	
	for Phrase in PhraseDict :
		PhraseDict[Phrase].NPQ = PhraseNPQ[Phrase]

	PhraseList = sorted(PhraseDict.items(), key = lambda d:d[1].NPQ, reverse = True)
	
	PhraseList = PhraseList[:len(PhraseList) * NeededPercentage // 100]
	PhraseDict.clear()
	for Phrase in PhraseList :
		PhraseDict[Phrase[1].Phrase] = Phrase[1]
	DeleteUnexisted(PhraseFrequency, PhraseDict)
	DeleteUnexisted(PhraseIDF, PhraseDict)
	DeleteUnexisted(PhraseNPQ, PhraseDict)

	print('Phrase\tNPQ\tIDF\tPMI\tCMI', file = Dic)
	for Phrase in PhraseList :
		print(Phrase[1].Phrase, '\t', Phrase[1].NPQ, '\t', Phrase[1].IDF, '\t', Phrase[1].PMI, '\t', Phrase[1].CMI, file = Dic)
	
	print('Phrase Abstractions...', file = Abs)
	AbstractDict = CalcAbstractTFIDF(DocDict, PhraseDict, RequiredTFIDFNum)
	for Doc in AbstractDict :
		print(Doc, AbstractDict[Doc], file = Abs)
	
	sys.stderr.write('In main() : Calculating Text Rank : ' + '\n')
	for Day in DocDict :
		Doc = DocDict[Day]
		PatternDateline = re.compile(r'(?<=#)[\s\S]*?(?=#)')
		Article = PatternDateline.findall('#' + Doc)
		print('Date ', Day, file = Rnk)
		sys.stderr.write('\t' + 'Data' + Day + '.\n')
		Count = 0
		for Line in Article :
			PhraseRankDict = T.TextRank(PhraseDict, DivideSet, Line, TextRankK, IterationRound, TextRankD, PenaltyConstant)
			PhraseRankList = sorted(PhraseRankDict.items(), key = lambda d:d[1], reverse = True)
			print('Article ', Count, ' : ', file = Rnk, end = '')
			for i in range(0, min(RequiredTextRankNum, len(PhraseRankList))) :
				print(PhraseRankList[i], file = Rnk, end = '')
			print('', file = Rnk)
			Count += 1
	Log.close()
	Dic.close()
	Abs.close()
	DataFile.close()

if __name__ == '__main__':
	main()