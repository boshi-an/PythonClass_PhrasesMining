import math
import re
import collections
import sys

class PhraseGraph :
	Edge = collections.defaultdict(list)
	Val = collections.defaultdict(int)
	InDegree = collections.defaultdict(int)
	OutDegree = collections.defaultdict(int)
	Vis = collections.defaultdict(int)
	AllPhrases = []
	BasicPhrases = []
	Stack = []

	def EraseSubgraph(self, Source, Amount, Root) :
		if self.Vis[Source] : 
			return

		if self.Val[Source] - Amount < 0 :
			print(Source, Root)
		self.Val[Source] = self.Val[Source] - Amount
		self.Vis[Source] = 1

		assert(self.Val[Source] >= 0)
		assert(Amount > 0)

		for to in self.Edge[Source] :
			self.EraseSubgraph(to, Amount, Root)
		
		if self.Val[Source] == 0 :
			for to in self.Edge[Source] :
				self.InDegree[to] = self.InDegree[to] - 1
				self.OutDegree[Source] = self.OutDegree[Source] - 1
				if self.InDegree[to]==0 and self.Val[to] :
					self.Stack.append(to)
		return
	
	def FindBasicPhrases(self) :
		for Phrase in self.AllPhrases :
			if (self.InDegree[Phrase] == 0) and self.Val[Phrase] :
				self.Stack.append(Phrase)
		while len(self.Stack) :
			Top = self.Stack.pop()
			print(Top, self.Val[Top])
			self.Vis.clear()
			self.EraseSubgraph(Top, self.Val[Top], Top)
			self.BasicPhrases.append(Top)
		return


	def __init__(self, FrequentPhrases) :
		self.Val = FrequentPhrases
		for Phrase in self.Val :
			self.Val[Phrase] = self.Val[Phrase] - 1
		for Phrase in FrequentPhrases :
			if len(Phrase) == 1 :
				self.AllPhrases.append(Phrase)
				continue
			Pre = Phrase[:-1]
			Suf = Phrase[1:]
			self.Edge[Phrase].append(Pre)
			self.Edge[Phrase].append(Suf)
			self.InDegree[Pre] = self.InDegree[Pre] + 1
			self.InDegree[Suf] = self.InDegree[Suf] + 1
			self.OutDegree[Phrase] = self.OutDegree[Phrase] + 1
			self.AllPhrases.append(Phrase)
		return
	

# Get phrases that appears more than $Threshold <int> in $Corpus <string>, and $DivideSet <set> means all the punctuation marks
def FrequentPhraseDetection(Corpus, Threshold, DivideSet) :
	Len = len(Corpus)
	Frequency = collections.defaultdict(int)
	Index  = collections.defaultdict(set)
	for i in range(Len) :
		if Corpus[i] not in DivideSet :
			Index[Corpus[i]].add(i)
	while len(Index) :
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

DayArticle = {}
#DayCorpus = ReadDayArticle("./Data/xin_cmn_200010")
DayCorpus = ReadDayArticle("./Data/test")
Corpus = ''
for Day in DayCorpus :
	Corpus = Corpus + DayCorpus[Day]
DivideSet = {'\n', ',', '・', '。', '“', '”', '(', ')', '?', '!', ':', '　', '―', '─', '、', ';', '-'}
FrequentPhrases = FrequentPhraseDetection(Corpus, 2, DivideSet)
sys.stderr.write('Frequent phrases estimated! ' + str(len(FrequentPhrases)) + ' Phrases.\n')
#for Phrase in FrequentPhrases :
#	print(Phrase, FrequentPhrases[Phrase])
Graph = PhraseGraph(FrequentPhrases)
Graph.FindBasicPhrases()
#for Phrase in Graph.BasicPhrases :
#	print(Phrase)