'''

Hosts global constants for the BTeventSelection.py code

'''

def BTcuts(dataset):

	BT_E_250                      = False
	
	if dataset == 'orbitMC' :
		DO_PMO_PRESELECTION           = False
		HIGH_REC_ENERGY_SELECTION     = True
		HIGH_ENERGY_TRIGGER_SELECTION = True
		REMOVE_PILEUP                 = False
		REMOVE_PILEUP_IMPACT          = False
		REMOVE_PILEUP_BGO             = False
		NO_BGO_CRACK_SELECTIONY       = False
		LONG_TRACK_SELECTION          = False
		NO_BGO_CRACK_SELECTIONX       = False
		INVERSE_NOCRACKSELECTION      = False
		HIGH_REC_ENERGY_MIN           = 1000000.   if args.startenergy is None else args.startenergy  #50000.  # PROTON / ELECTRON HIGH ENERGY (250 GeV)
		HIGH_REC_ENERGY_MAX           = 1300000.   if args.stopenergy is None else args.stopenergy    #150000. # PROTON / ELECTRON HIGH ENERGY (250 GeV)
	
	elif dataset == 'protonBT' :
		DO_PMO_PRESELECTION           = True
		HIGH_REC_ENERGY_SELECTION     = True
		HIGH_ENERGY_TRIGGER_SELECTION = True
		REMOVE_PILEUP                 = True
		REMOVE_PILEUP_IMPACT          = False
		REMOVE_PILEUP_BGO             = True
		NO_BGO_CRACK_SELECTIONY       = False
		LONG_TRACK_SELECTION          = False
		NO_BGO_CRACK_SELECTIONX       = False
		INVERSE_NOCRACKSELECTION      = False
		HIGH_REC_ENERGY_MIN           = 150000.
		HIGH_REC_ENERGY_MAX           = 99999999.
	
	elif dataset == 'electron250' :
		DO_PMO_PRESELECTION           = True		# Rejects 4480 events
		HIGH_REC_ENERGY_SELECTION     = True		# Rejects 2500 events (10%)
		HIGH_ENERGY_TRIGGER_SELECTION = True		# Rejects 1438 events
		REMOVE_PILEUP                 = True		
		REMOVE_PILEUP_IMPACT          = False
		REMOVE_PILEUP_BGO             = True		# Rejects 64 events
		NO_BGO_CRACK_SELECTIONY       = True		# all
		LONG_TRACK_SELECTION          = True		# all
		NO_BGO_CRACK_SELECTIONX       = True		# all
		INVERSE_NOCRACKSELECTION      = False
		HIGH_REC_ENERGY_MIN           = 150000.   #50000.  # PROTON / ELECTRON HIGH ENERGY (250 GeV)
		HIGH_REC_ENERGY_MAX           = 99999999. #150000. # PROTON / ELECTRON HIGH ENERGY (250 GeV)
		BT_E_250                      = True

	elif dataset == 'electron250i' :			# Inclined
		DO_PMO_PRESELECTION           = True
		HIGH_REC_ENERGY_SELECTION     = True
		HIGH_ENERGY_TRIGGER_SELECTION = True
		REMOVE_PILEUP                 = True
		REMOVE_PILEUP_IMPACT          = False
		REMOVE_PILEUP_BGO             = False
		NO_BGO_CRACK_SELECTIONY       = False
		LONG_TRACK_SELECTION          = True
		NO_BGO_CRACK_SELECTIONX       = True
		INVERSE_NOCRACKSELECTION      = False
		HIGH_REC_ENERGY_MIN           = 150000.   #50000.  # PROTON / ELECTRON HIGH ENERGY (250 GeV)
		HIGH_REC_ENERGY_MAX           = 99999999. #150000. # PROTON / ELECTRON HIGH ENERGY (250 GeV)
		BT_E_250                      = True

	elif dataset == 'electron100' :
		DO_PMO_PRESELECTION           = True
		HIGH_REC_ENERGY_SELECTION     = True
		HIGH_ENERGY_TRIGGER_SELECTION = True
		REMOVE_PILEUP                 = True
		REMOVE_PILEUP_IMPACT          = False
		REMOVE_PILEUP_BGO             = True
		NO_BGO_CRACK_SELECTIONY       = True
		LONG_TRACK_SELECTION          = True
		NO_BGO_CRACK_SELECTIONX       = True
		INVERSE_NOCRACKSELECTION      = False
		HIGH_REC_ENERGY_MIN           = 50000.   #50000.  # PROTON / ELECTRON HIGH ENERGY (250 GeV)
		HIGH_REC_ENERGY_MAX           = 150000.  #150000. # PROTON / ELECTRON HIGH ENERGY (250 GeV)

	elif dataset == 'electron100i' :	# inclined
		DO_PMO_PRESELECTION           = True
		HIGH_REC_ENERGY_SELECTION     = True
		HIGH_ENERGY_TRIGGER_SELECTION = True
		REMOVE_PILEUP                 = True
		REMOVE_PILEUP_IMPACT          = False
		REMOVE_PILEUP_BGO             = False
		NO_BGO_CRACK_SELECTIONY       = False
		LONG_TRACK_SELECTION          = True
		NO_BGO_CRACK_SELECTIONX       = True
		INVERSE_NOCRACKSELECTION      = False
		HIGH_REC_ENERGY_MIN           = 50000.   #50000.  # PROTON / ELECTRON HIGH ENERGY (250 GeV)
		HIGH_REC_ENERGY_MAX           = 200000.  #150000. # PROTON / ELECTRON HIGH ENERGY (250 GeV)


	"""
	# ELECTRON CUTS
	NO_BGO_CRACK_SELECTIONY       = True
	LONG_TRACK_SELECTION          = True
	NO_BGO_CRACK_SELECTIONX       = True # 100 Gev 2014 electrons
	#
	INVERSE_NOCRACKSELECTION      = True
	"""
	
	return DO_PMO_PRESELECTION, HIGH_REC_ENERGY_SELECTION, HIGH_ENERGY_TRIGGER_SELECTION, REMOVE_PILEUP, REMOVE_PILEUP_IMPACT, REMOVE_PILEUP_BGO, NO_BGO_CRACK_SELECTIONY, LONG_TRACK_SELECTION, NO_BGO_CRACK_SELECTIONX, INVERSE_NOCRACKSELECTION, HIGH_REC_ENERGY_MIN, HIGH_REC_ENERGY_MAX, BT_E_250


###################################
###################################
###################################
###################################


####
# Global constants
##

NBGOLAYERS  = 14
NBARSLAYER  = 22
EDGEBARS    = [0,21]
BARPITCH    = 27.5
PLANESATBGO = [3,4,5]
PLANESATPSD = [0]
PLANESATBGO = [3,4,5]
PLANEATBGO  = 5
F_LAYER = 13

NEVENTS = None

DO_XTRL = True

COARSEBIN = False

PLOT_BGO_NOISE_HISTOGRAMS     = True

PSD_DOUBLEMIP_CUT             = False
TRACK_SELECTION               = False
TRACK_NOTPROTON_SELECTION     = False
TRACK_MINNPOINTS_SELECTION    = 4

# HISTOGRAM FOR 2 TYPE OF EVENTS
HISTO_XTR_THRESHOLD = 12.
PSD_HIT_ENERGY_CUT  = 1.0
BGO_STK_MATCH_CUT   = 0.1
# DOUBLE MIP CUT
PSD_HIT_ENERGY_DOUBLEMIP_CUT  = 2.0

# TRACK HISTOGRAMS
DO_TRACK_HISTOS = True

# TRACK CUTS
CHARGE_AVERAGE_CUT = 100.

# Andrii - for PMO selection 
FIRSTLAYERSNOTSIDE = [1,2,3]
MAXLAYERCUT = 0.35
BGO_BAR_STEP = 27.5
BGO_CRACK_DISTANCE = 1.25
BGO_CRACK_ADDSAFETYMARGIN = 2. # 6.
BGO_STK_INTERCEPTMATCH = 50.
BGO_STK_ANGMATCH = 0.15 #0.07

PILEUP_DISTANCE    = 20.
PILEUP_BGO_Z       = -210.
PILEUP_BGO_XYCUT   = 25    #40.
PILEUP_BGO_INCLCUT = 0.11  #0.15

PILEUP_ZCOORDINATES = [ 
			-210, 
			-176, 
			-144, 
			-111, 
			-79 , 
			-46 , 
			]


