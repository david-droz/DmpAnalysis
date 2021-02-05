

class Selecter(object):
	
	def __init__(self,pev):
		
		
		### DOES IT EVEN MAKE SENSE TO WRITE THIS CLASS?
		# Can simply add the manual RMS/XTRL computation to the "selection.py" code and that's it ... right?
		# Have to find where I did this computation tho  ---> selection_Simone.py
		
		
		
		_checkEvent(pev)
		
	def _checkEvent(pev):
		
		[...]
		
		self.isGood = True
		
	def isGoodEvent():
		return self.isGood 


def getXTRL(pev):
	
	NBGOLAYERS  = 14
	NBARSLAYER  = 22
	EDGEBARS    = [0,21]
	BARPITCH    = 27.5
	
	edep = np.zeros((NBGOLAYERS,NBARSLAYER))
	for i in xrange(NBGOLAYERS):
		for j in xrange(NBARSLAYER):
			edep[i,j] = pev.pEvtBgoRec().GetEdep(i,j)
	
	BHET = edep.sum()
	BHXS = [0. for i in xrange(NBGOLAYERS)]
	BHER = [0. for i in xrange(NBGOLAYERS)]
	COG = [0. for i in xrange(NBGOLAYERS)]
	bhm  = 0.
	SIDE = [False for i in xrange(NBGOLAYERS)]
	
	for i in xrange(NBGOLAYERS):
		# Find the bar with max energy deposition of a layer and record its number as im
		im = None
		em = 0.0;
		for j in xrange(NBARSLAYER):
			ebar = edep[i,j]
			if ebar < em : continue 
			em = ebar
			im = j
		
		if not em: continue
		
		if im in EDGEBARS:
			cog = BARPITCH * im   #BHX[i][im]
			
		else:
			ene = 0.
			cog = 0.
			for  j in [im-1, im, im+1]: 
				ebar = edep[i,j]
				ene += ebar
				cog += BARPITCH * j * ebar
			cog /= ene
			
		posrms   = 0.
		enelayer = 0.
		for j in xrange(NBARSLAYER):
			ebar = edep[i,j]
			posbar = BARPITCH * j 
			enelayer += ebar
			posrms += ebar * (posbar-cog)*(posbar-cog)
		posrms = math.sqrt( posrms / enelayer)
		BHXS[i] = posrms
		COG[i] = cog
		BHER[i] = enelayer / BHET
	
	sumRMS = sum(BHXS)
	F = [r for r in reversed(BHER) if r][0]
	XTRL = sumRMS**4.0 * F / 8000000.
	
	del edep
	
	return BHER, BHXS, XTRL
