from collections import defaultdict
import time     #for sleep
from Bio import Entrez
import requests #for get requests
import xml.etree.ElementTree as ET

__author__ = "Jayse Farrell, Jessica Kunder, Ryan Palm"
__copyright__ = "COPYRIGHT_INFORMATION"
__credits__ = ["Jayse Farrell, Jessica Kunder, Ryan Palm"]
__license__ = "GPL"
__version__ = "1.6.0dev"
__maintainer__ = "AUTHOR_NAME"
__email__ = "AUTHOR_EMAIL"
__status__ = "homework"

baseURL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
#https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&id=2099&retmode=xml
baseURL2 = "https://ncbi.nlm.nih.gov"
Tag = "{http://www.w3.org/1999/xhtml}"
divID = "viewercontent1"

def main():
    Entrez.email = "jaysef2@uw.edu"
    Entrez.tool = "FetchFASTA"
    IDs = FetchID()
    URLs = defaultdict(str)

    for ID in IDs:
        temp = FetchURL(id=ID)
        print(ID, temp)
        if temp != None:
            URLs[ID] = temp

    file_object = open('fasta_contents.txt', 'w')
    name_object = open('species_name.txt', 'w')

    print(URLs.values())
    for name, content in FetchFASTA(URLs.values()): #uses generator to retrieve name/content pairs, writes them to files
        name_object.write(name + '\n')
        file_object.write(content + '\n')
        print(content)

    #closes files
    file_object.close()
    name_object.close()

def FetchID():
    #uses the Entrez API to recieve a list of gene ids associate with the EGR1 gene
    search = Entrez.read(Entrez.esearch(db="gene", retmax=250, term="EGR1[gene]")) #retrieves ID's for all stored ESR1 genes
    # print(search['IdList'])
    return search["IdList"]

def FetchURL(id=None):
    if id is None:
        return None

    #recieves xml code for the webpage of a given gene
    httpRequest = requests.get(baseURL2 + '/gene/' + id)

    #checks if request completed successfully
    if httpRequest.status_code != 200:
        print('bad request')
        return None

    #builds an element tree
    root = ET.fromstring(httpRequest.text)

    #recieves the a tag with a title of Nucleotide FASTA report
    # for child in root.iter(tag=(Tag + 'a')):
    #     if child.get('title') != None and child.get('title') == 'Nucleotide FASTA report':
    #         print(child, child.get('title'))
    childNode = [x for x in root.iter(tag=(Tag + 'a')) if x.get('title') != None and x.get('title') == 'Nucleotide FASTA report']

    if(len(childNode) == 0):
        return None
    childNode = childNode[0]

    #returns href if it has one, otherwise prints error message
    if childNode.get("href") != None:
        return childNode.get("href")
    else:
        print("no child Node for ", id)

    return None

def FetchFASTA(urls=None):
    print('entered')
    for url in urls:
        #extracts the nucleotid id, the start point, and the end point from the given url
        id = find_between(url, '/nuccore/', '?')
        start = find_between(url, 'from=', '&to')
        end = find_between(url, 'to=', '&strand=')
        if end == None:
            end = url[url.index('to=') + 3:]

        #using the information attained above, request the fasta file associated with it using the Entrez API
        fetch = Entrez.efetch(db='nuccore', id=id,
        seq_start=start, seq_stop=end, rettype='fasta', tetmode='text')

        #combines lines into a single string, extracts name and condenses opening tag
        content = ""
        name = ''
        i = False
        for temp in fetch:
            if i == False:
                content += '> ' + find_between(temp, ' ', ',') + '\n'
                name = find_between(temp, ' ', ',')
                i = True
            else:
                content += temp;
        yield name, content.strip()

def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        if start > end:
            return None
        return s[start:end]
    except ValueError:
        return None
main()
