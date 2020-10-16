import math
import re
import collections
import sys
from queue import Queue
from sklearn.ensemble import RandomForestClassifier
import random

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
				if len(self.Endp[x]) :
					Occured[self.Endp[x]] += 1
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
	return pow(Length, Constant)

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
	return

def main() :
	PenaltyConstant = 1.1
	NeededPercentage = 25
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
	DataFile = open('./TrainingData.txt', 'w')
	Corpus = ''
	DocDict = collections.defaultdict(str)
	PhraseDict = collections.defaultdict(PhraseProperties)

	for Date in DateList :
		DayCorpus = ReadDayArticle('./Data/' + Date)
		DocDict.update(DayCorpus)
		for Day in DayCorpus :
			Corpus = Corpus + '.' + DayCorpus[Day]
	PhraseFrequency = FrequentPhraseDetection(Corpus, 10, DivideSet)
	sys.stderr.write('Frequent phrases estimated! ' + str(len(PhraseFrequency)) + ' Phrases.\n')
	
	#for Phrase in PhraseFrequency :
	#	print(Phrase, PhraseFrequency[Phrase], file = Log)

	PhraseRFC = collections.defaultdict(float)
	PhrasePMI = collections.defaultdict(float)
	PhraseCMI = collections.defaultdict(float)
	PhrasePKL = collections.defaultdict(float)
	PhraseIDF = collections.defaultdict(float)

	Sum = 0
	for Phrase in PhraseFrequency :
		Sum += PhraseFrequency[Phrase]
	for Phrase in PhraseFrequency :
		PhraseRFC[Phrase] = 1.0 * PhraseFrequency[Phrase] / Sum
	
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
	
	PhraseList = []
	for Phrase in PhraseDict : PhraseList.append(Phrase)
	random.shuffle(PhraseList)
	for Phrase in PhraseList[:300] :
		a = input(Phrase) or 0
		print(a,\
					Phrase,\
					PhraseDict[Phrase].RFC,\
					PhraseDict[Phrase].PMI,\
					PhraseDict[Phrase].CMI,\
					PhraseDict[Phrase].PKL,\
					PhraseDict[Phrase].IDF,\
					file = DataFile)

	Log.close()
	Dic.close()
	Abs.close()
	DataFile.close()

if __name__ == '__main__':
    main()