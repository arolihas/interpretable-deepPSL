nominal_set = {'Mitochondrion', 'Peroxisome', 'Plastid', 'Cytoplasm', 'Extracellular'}
nuclear_list = ['Nucleus', 'Chromosome', 'nucleus']
er_list = ['Endoplasmic reticulum', 'endoplasmic reticulum', 'Microsome', 'Sarcoplasmic reticulum']
golgi_list = ['Golgi apparatus', 'golgi apparatus']
lyso_list = ['contractile', 'vacuole', 'lysosome']
membrane_list = ['Apical', 'apicolateral', 'basal', 'basolateral',
 'lateral', 'cell membrane', 'cell projection']
extra_set = {
    'Nucleus': nuclear_list,
    'Endoplasmic.reticulum': er_list,
    'Golgi.apparatus': golgi_list,
    'Cell.membrane': membrane_list,
    'Lysosome/Vacuole': lyso_list
}
def determine_locations(subcell, subclass):
    if subclass in nominal_set:
        return 1 if subclass in subcell else 0
    else:
        possible_classes = extra_set[subclass]
        if subcell in possible_classes:
            return 1
        return int(any([subcell.contains(c) for c in possible_classes]))