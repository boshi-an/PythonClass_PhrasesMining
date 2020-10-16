import math
import re
import collections
import sys
from queue import Queue
from sklearn.ensemble import RandomForestClassifier
import random
import threading

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

class MyThread(threading.Thread) :
	def __init__(self, func, args=()) :
		threading.Thread.__init__(self)
		self.func = func
		self.args = args

	def run(self) :
		self.result = self.func(*self.args)
	
	def get_result(self):
		try:
			return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
		except Exception:
			return None

class AhoCorasickAutomaton :
	def Insert(self, Str) :
		x = 1
		for c in Str :
			if self.Next[x][c] == 0 :
				self.Size += 1
				self.Next.append(collections.defaultdict(int))
				self.Endp.append('')
				self.Fail.append(0)
				self.Next[x][c] = self.Size
			x = self.Next[x][c]
		self.Endp[x] = Str
		return
	
	def Build(self) :
		q = Queue(maxsize = 0)
		self.Fail[1] = 1
		for c in self.Next[1] :
			y = self.Next[1][c]
			self.Fail[y] = 1
			q.put(y)
		while q.qsize() :
			x = q.get()
			for c in self.Next[x] :
				y = self.Next[x][c]
				self.Fail[y] = self.Next[self.Fail[x]][c]
				q.put(y)
		return

	def Match(self, Str) :
		Occured = collections.defaultdict(int)
		x = 1
		for c in Str :
			while (x!=1) and (self.Next[x][c]==0) :
				x = self.Fail[x]
			if self.Next[x][c] :
				x = self.Next[x][c]
				y = x
				while y != 1 :
					if len(self.Endp[y]) :
						Occured[self.Endp[y]] += 1
					y = self.Fail[y]
		return Occured
		
	def __init__(self, StrList) :
		self.Size = 1
		self.Next = [collections.defaultdict(int), collections.defaultdict(int)]
		self.Endp = ['', '']
		self.Fail = [1, 1]
		for Str in StrList :
			self.Insert(Str)
		self.Build()
		return
		

def LengthPenalty(Constant, Length) :
	return pow(Length, Constant) - 1

def DeleteUnexisted(Dict, Model) :
	Copy = Dict.copy()
	for Phrase in Copy :
		if Phrase not in Model :
			Dict.pop(Phrase)

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
		for j in range(1, i+1) :
			Cur = Corpus[i-j] + Cur
			if Cur not in NPQ :
				break
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
		DayCorpus[Date] = DayCorpus[Date] + Corpus
	return DayCorpus

# Calculate Inverse Document Frequency
def CalcIDF(DocDict, PhraseFrequency) :
	sys.stderr.write('In CalcIDF() : Calculating IDF .\n')
	AC = AhoCorasickAutomaton(PhraseFrequency)
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

def CalcAbstract(DocDict, PhraseDict) :
	sys.stderr.write('In CalcAbstract() : Calculating abstract phrases for each document .\n')
	Abstract = collections.defaultdict(list)
	AC = AhoCorasickAutomaton(PhraseDict)
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
		Abstract[Doc] = PossibleAbstractList[:min(25, len(PossibleAbstractList))]
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

def main() :
	IterationRound = 30
	FrequencyLimit = 10
	PenaltyConstant = 1.1
	NeededPercentage = 25
	MaxThread = 2
	ThreadBlock = 1024
	assert(ThreadBlock % MaxThread == 0)
	DateList = [
		'xin_cmn_200010',
		#'xin_cmn_200011',
		#'xin_cmn_200012',
		#'xin_cmn_200101',
		#'xin_cmn_200304',
		#'xin_cmn_200305',
		#'xin_cmn_200306',
		#'xin_cmn_200307',
		#'xin_cmn_200308',
		#'xin_cmn_200309'
	]
	DivideSet = {',', '・', '。', '“', '”', '(', ')', '?', '!', ':', '　', '―', '─', '、', ';', '-', ' ', '》', '《', '.'}
	Log = open('./Log.txt', 'w')
	Dic = open('./Dic.txt', 'w')
	Abs = open('./Abs.txt', 'w')
	DataFile = open('./TrainingData.txt', 'r')
	Correction = open('./Correct.txt', 'w')
	Corpus = ''
	DocDict = collections.defaultdict(str)
	PhraseDict = collections.defaultdict(PhraseProperties)

	for Date in DateList :
		DayCorpus = ReadDayArticle('./Data/' + Date)
		DocDict.update(DayCorpus)
		for Day in DayCorpus :
			Corpus = Corpus + '.' + DayCorpus[Day]
	PhraseFrequency = FrequentPhraseDetection(Corpus, FrequencyLimit, DivideSet)
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
		PhrasePMI[Phrase] = -1000
		PhraseCMI[Phrase] = -1000
		PhrasePKL[Phrase] = -1000
		PhraseIDF[Phrase] = -1000

	for Phrase in PhraseFrequency :
		Len = len(Phrase)
		for i in range(1, Len) :
			u = Phrase[:i]
			v = Phrase[i:]
			PhrasePMI[Phrase] = max(PhrasePMI[Phrase], math.log(PhraseRFC[Phrase] / PhraseRFC[u] / PhraseRFC[v]))
			PhraseCMI[u] = max(PhraseCMI[u], math.log(PhraseRFC[Phrase] * PhraseRFC[Phrase] / PhraseRFC[u] / PhraseRFC[v]))
			PhraseCMI[v] = max(PhraseCMI[v], math.log(PhraseRFC[Phrase] * PhraseRFC[Phrase] / PhraseRFC[u] / PhraseRFC[v]))
			PhrasePKL[Phrase] = max(PhrasePKL[Phrase], PhraseRFC[Phrase] * PhrasePMI[Phrase])
	PhraseIDF = CalcIDF(DocDict, PhraseFrequency)
	
	for Phrase in PhraseFrequency :
		PhraseDict[Phrase] = PhraseProperties(Phrase,\
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
		Phrase = Splited[1]
		Correction.write(str(Splited[0]) + ' ' + Phrase + ' ' + str(PhraseRFC[Phrase]) + ' ' + str(PhrasePMI[Phrase]) + ' ' + str(PhraseCMI[Phrase]) + ' ' + str(PhrasePKL[Phrase]) + ' ' + str(PhraseIDF[Phrase]) + '\n')
	
	quit()

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
		Cut = CutCorpus(Corpus, PenaltyConstant, PhraseNPQ)
		PhraseNPQ = ReCalculateNPQ(Corpus, Cut, PenaltyConstant, PhraseFrequency, PhraseNPQ)
	
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
	AbstractDict = CalcAbstract(DocDict, PhraseDict)
	for Doc in AbstractDict :
		print(Doc, AbstractDict[Doc], file = Abs)

	Log.close()
	Dic.close()
	Abs.close()
	DataFile.close()

if __name__ == '__main__':
	main()