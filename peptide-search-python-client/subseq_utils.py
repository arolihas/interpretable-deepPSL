from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from bioservices import UniProt
from pprint import pprint
import pandas as pd

def peptide_search(peptides, swissprot=True, isoform=True):
    api_instance = swagger_client.PeptideMatchAPI20Api()
    api_response = None
    while api_response == None:
        try:
            # Do peptide match using GET method. Default taxonomic ids human and mice
            api_response = api_instance.match_get_get(peptides, swissprot=swissprot, isoform=isoform)
        except ApiException as e:
            print("Exception when calling PeptideMatchAPI20Api->match_get_get: %s\n" % e)
            return None
    return api_response.results[0].proteins

def hitrate(proteins, indexes, subclass):
    columns = ['subsequence', 'sprot_start', 'sprot_end', 'sprot_loc', 'dl_start', 'dl_end', 'dl_loc']
    u = UniProt()
    match, total = 0,0
    dl_peptides, dl_starts, dl_ends, sprot_starts, sprot_ends, sprot_locs =[], [], [], [], [], []
    if proteins != None:
        for prot in proteins:
            locs = None
            try:
                entry = u.retrieve(prot.ac, frmt='xml')
                locs = entry['subcellularlocation']
            except:
                continue
            if locs:
                pep_metadata = prot.matching_peptide[0]
                seq_range = pep_metadata.match_range[0]
                peptide = pep_metadata.peptide
                dl_peptides.append(peptide)
                start, end = indexes[peptide]
                dl_starts.append(start)
                dl_ends.append(end)
                
                pos = seq_range.start
                sprot_starts.append(pos)
                sprot_ends.append(seq_range.end)
                
                seq_len = seq_range.end - pos
                offset_weight = 1 if pos == start else min(abs(seq_len / (pos - start)), 1) 
                
                loc = list((locs[0].children))[1].string
                sprot_locs.append(loc)
                match_weight = determine_locations(loc, subclass) * offset_weight
                match += match_weight
                # assert(match_weight <= 1),'match_weight {}'.format(match_weight)
                total += offset_weight
    if total == 0:
        hitrate = 0
    else:
        hitrate = match/total
    vals = [[dl_peptides, sprot_starts, sprot_ends, sprot_locs, dl_starts, dl_ends, subclass]]
    df = pd.DataFrame(vals, columns=columns)
    return (hitrate, df)

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
        return int(any([c in subcell for c in possible_classes]))