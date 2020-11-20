from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint
from bioservices import UniProt
import subcellclasses as sc

# create an instance of the API class
api_instance = swagger_client.PeptideMatchAPI20Api()
peptides = 'MAAAAGR' # str | A list of comma-separated peptide sequences (up to 100). Each sequence consists of 3 or more amino acids. (default to AAVEEGIVLGGGCALLR,SVQYDDVPEYK)
peptide_index = 1
taxonids = '' #'9606,10090' # str | A list fo comma-separated NCBI taxonomy IDs. (optional) (default to 9606,10090)
swissprot = True # bool | Only search SwissProt protein sequences. (optional) (default to true)
isoform = True # bool | Include isforms. (optional) (default to true)

try:
    # Do peptide match using GET method.
    api_response = api_instance.match_get_get(peptides, taxonids=taxonids, swissprot=swissprot, isoform=isoform)
except ApiException as e:
    print("Exception when calling PeptideMatchAPI20Api->match_get_get: %s\n" % e)

u = UniProt()
print("Peptide matches for {} at position {}".format(peptides, peptide_index))
proteins = api_response.results[0].proteins
match, total = 0, 0
for result in proteins:
    res = u.retrieve(result.ac, frmt='xml')
    locs = res['subcellularlocation']
    if locs:
        loc = list((locs[0].children))[1].string
        pos = result.matching_peptide[0].match_range[0].start
        if pos==peptide_index:
            print("Entry {} : {}".format(result.ac, loc))
            match += sc.determine_locations(loc, 'Mitochondrion')
            total += 1
        else:
            offset_weight = min(len(peptides) / (pos - peptide_index), 1)
            print("Entry {} : {} at position {}".format(result.ac, loc, pos))
            match += sc.determine_locations(loc, 'Mitochondrion') * offset_weight
            total += offset_weight
print("hitrate: {}".format(match/total))