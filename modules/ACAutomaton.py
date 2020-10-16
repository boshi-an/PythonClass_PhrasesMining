import collections
from queue import Queue

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