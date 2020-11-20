# swagger_client.PeptideMatchAPI20Api

All URIs are relative to *http://research.bioinformatics.udel.edu/peptidematchapi2*

Method | HTTP request | Description
------------- | ------------- | -------------
[**match_get_get**](PeptideMatchAPI20Api.md#match_get_get) | **GET** /match_get | Do peptide match using GET method.
[**match_post_post**](PeptideMatchAPI20Api.md#match_post_post) | **POST** /match_post | Do peptide match using POST method.


# **match_get_get**
> Report match_get_get(peptides, taxonids=taxonids, swissprot=swissprot, isoform=isoform, uniref100=uniref100, leqi=leqi, offset=offset, size=size)

Do peptide match using GET method.

Retrieve UniProtKB protein sequences that would exactly match to the query peptides using GET method.

### Example 
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.PeptideMatchAPI20Api()
peptides = 'AAVEEGIVLGGGCALLR,SVQYDDVPEYK' # str | A list of comma-separated peptide sequences (up to 100). Each sequence consists of 3 or more amino acids. (default to AAVEEGIVLGGGCALLR,SVQYDDVPEYK)
taxonids = '9606,10090' # str | A list fo comma-separated NCBI taxonomy IDs. (optional) (default to 9606,10090)
swissprot = true # bool | Only search SwissProt protein sequences. (optional) (default to true)
isoform = true # bool | Include isforms. (optional) (default to true)
uniref100 = false # bool | Only search UniRef100 protein sequences. (optional) (default to false)
leqi = false # bool | Treat Leucine (L) and Isoleucine (I) equivalent. (optional) (default to false)
offset = 0 # int | Off set, page starting point, with default value 0. (optional) (default to 0)
size = 100 # int | Page size with default value 100. When page size is -1, it returns all records and offset will be ignored. (optional) (default to 100)

try: 
    # Do peptide match using GET method.
    api_response = api_instance.match_get_get(peptides, taxonids=taxonids, swissprot=swissprot, isoform=isoform, uniref100=uniref100, leqi=leqi, offset=offset, size=size)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling PeptideMatchAPI20Api->match_get_get: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **peptides** | **str**| A list of comma-separated peptide sequences (up to 100). Each sequence consists of 3 or more amino acids. | [default to AAVEEGIVLGGGCALLR,SVQYDDVPEYK]
 **taxonids** | **str**| A list fo comma-separated NCBI taxonomy IDs. | [optional] [default to 9606,10090]
 **swissprot** | **bool**| Only search SwissProt protein sequences. | [optional] [default to true]
 **isoform** | **bool**| Include isforms. | [optional] [default to true]
 **uniref100** | **bool**| Only search UniRef100 protein sequences. | [optional] [default to false]
 **leqi** | **bool**| Treat Leucine (L) and Isoleucine (I) equivalent. | [optional] [default to false]
 **offset** | **int**| Off set, page starting point, with default value 0. | [optional] [default to 0]
 **size** | **int**| Page size with default value 100. When page size is -1, it returns all records and offset will be ignored. | [optional] [default to 100]

### Return type

[**Report**](Report.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json, application/xml, text/x-fasta, text/tab-separated-values

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **match_post_post**
> Report match_post_post(peptides, taxonids=taxonids, swissprot=swissprot, isoform=isoform, uniref100=uniref100, leqi=leqi, offset=offset, size=size)

Do peptide match using POST method.

Retrieve UniProtKB protein sequences that would exactly match to the query peptides using POST method.

### Example 
```python
from __future__ import print_function
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# create an instance of the API class
api_instance = swagger_client.PeptideMatchAPI20Api()
peptides = 'AAVEEGIVLGGGCALLR,SVQYDDVPEYK' # str | A list of comma-separated peptide sequences (up to 100). Each sequence consists of 3 or more amino acids. (default to AAVEEGIVLGGGCALLR,SVQYDDVPEYK)
taxonids = '9606,10090' # str | A list fo comma-separated NCBI taxonomy IDs. (optional) (default to 9606,10090)
swissprot = true # bool | Only search SwissProt protein sequences. (optional) (default to true)
isoform = true # bool | Include isoforms. (optional) (default to true)
uniref100 = false # bool | Only search UniRef100 protein sequences. (optional) (default to false)
leqi = false # bool | Treat Leucine (L) and Isoleucine (I) equivalent. (optional) (default to false)
offset = 0 # int | Off set, page starting point, with default value 0. (optional) (default to 0)
size = 100 # int | Page size with default value 100. When page size is -1, it returns all records and offset will be ignored. (optional) (default to 100)

try: 
    # Do peptide match using POST method.
    api_response = api_instance.match_post_post(peptides, taxonids=taxonids, swissprot=swissprot, isoform=isoform, uniref100=uniref100, leqi=leqi, offset=offset, size=size)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling PeptideMatchAPI20Api->match_post_post: %s\n" % e)
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **peptides** | **str**| A list of comma-separated peptide sequences (up to 100). Each sequence consists of 3 or more amino acids. | [default to AAVEEGIVLGGGCALLR,SVQYDDVPEYK]
 **taxonids** | **str**| A list fo comma-separated NCBI taxonomy IDs. | [optional] [default to 9606,10090]
 **swissprot** | **bool**| Only search SwissProt protein sequences. | [optional] [default to true]
 **isoform** | **bool**| Include isoforms. | [optional] [default to true]
 **uniref100** | **bool**| Only search UniRef100 protein sequences. | [optional] [default to false]
 **leqi** | **bool**| Treat Leucine (L) and Isoleucine (I) equivalent. | [optional] [default to false]
 **offset** | **int**| Off set, page starting point, with default value 0. | [optional] [default to 0]
 **size** | **int**| Page size with default value 100. When page size is -1, it returns all records and offset will be ignored. | [optional] [default to 100]

### Return type

[**Report**](Report.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/x-www-form-urlencoded
 - **Accept**: application/json, application/xml, text/x-fasta, text/tab-separated-values

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

