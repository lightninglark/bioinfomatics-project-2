import requests
import xml.etree.ElementTree as ET
import csv
from bs4 import BeautifulSoup
from collections import namedtuple

fileName = "species_name_temp.txt"
baseURL = "http://animaldiversity.org/accounts/"
# dataFile = 'finalData.csv'


class DesiredData:

    characteristic = namedtuple('characteristic', 'isInitialized value')

    def __init__(self, name):
        self.name = name
        self.monogamous = namedtuple('monogamous', 'isInitialized value')
        self.polygynous = namedtuple('polygynous', 'isInitialized value')
        self.polyandrous = namedtuple('polyandrous', 'isInitialized value')
        self.promiscuous = namedtuple('promiscuous', 'isInitialized value')
        self.solitary = namedtuple('solitary', 'isInitialized value')
        self.small_units = namedtuple('small_units', 'isInitialized value')
        self.communal = namedtuple('communal', 'isInitialized value')
        self.parental_investment = namedtuple('parental_investment', 'isInitialized value')
        self.noParental_investment = namedtuple('noParental_investment', 'isInitialized value')
        self.caste_system = namedtuple('caste_system', 'isInitialized value')
        self.noCaste_system = namedtuple('noCaste_system', 'isInitialized value')

        self.monogamous.isInitialized = self.polygynous.isInitialized = self.polyandrous.isInitialized = \
            self.promiscuous.isInitialized = self.solitary.isInitialized = self.small_units.isInitialized = \
            self.communal.isInitialized = self.parental_investment.isInitialized = self.noParental_investment.isInitialized = \
            self.caste_system.isInitialized = self.noCaste_system.isInitialized = False
        self.monogamous.value = self.polygynous.value = self.polyandrous.value = \
            self.promiscuous.value = self.solitary.value = self.small_units.value = \
            self.communal.value = self.parental_investment.value = self.noParental_investment.value = \
            self.caste_system.value = self.noCaste_system.value = False
        # print(str(self.monogamous.isInitialized))

    def getMonogamous(self):
        return self.monogamous.value
    def getPolygynous(self):
        return self.polygynous.value
    def getPolyandrous(self):
        return self.polyandrous.value
    def getPromiscuous(self):
        return self.promiscuous.value
    def getSolitary(self):
        return self.solitary.value
    def getSmall(self):
        return self.small_units.value
    def getCommunal(self):
        return self.communal.value
    def getParental(self):
        return self.parental_investment.value
    def getNonParental(self):
        return self.noParental_investment.value
    def getCaste(self):
        return self.caste_system.value
    def getNonCaste(self):
        return self.noCaste_system.value


def main():
    with open(fileName, "r") as species_names:
        with open('finalData.csv', 'w') as dataFile:
            writer = csv.writer(dataFile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['Name', 'Monogamous', 'Polygynous', 'Polyandrous', 'Promiscuous', 'Solitary',
                             'Small Units', 'communal', 'parental investment',
                             'no parental investment', 'Caste System', 'no Caste System'])
            for line in species_names:
                data = DesiredData(line.strip().replace(' ', '_'))
                element_tree = getelementtree(baseURL + data.name)
                if element_tree is None:
                    continue
                else:
                    # print(str(data.monogamous.isInitialized))
                    # print(str(data.monogamous.value))
                    extract_mating(element_tree, data)
                    # print(str(data.monogamous.isInitialized))
                    # print(str(data.getMonogamous()))
                    row = [data.name, str(data.getMonogamous()) if data.monogamous.isInitialized else None,
                           str(data.getPolygynous()) if data.polygynous.isInitialized else None,
                           str(data.getPolyandrous()) if data.polyandrous.isInitialized else None,
                           str(data.getPromiscuous()) if data.promiscuous.isInitialized else None]
                    writer.writerow(row)
                    print(data.name)
            dataFile.close()


def extract_mating(soup, data):
    # ListNode = [p for p in soup.find_all('a')]
    node = [child for child in soup.find_all() if child.get_text() == 'Mating System']
    if len(node) > 0:
        parent = node[0].find_parent('ul')
        characteristics = parent.find_all('li')
        for child in characteristics:
            text = child.get_text().strip()
            if text == 'monogamous':
                data.monogamous.value = True
            elif text == 'polyandrous':
                data.polyandrous.value = True
            elif text == 'polygynous':
                data.polygynous.value = True
            elif text == 'polygynandrous (promiscuous)':
                data.promiscuous.value = True
            else:
                continue
        data.monogamous.isInitialized = True
        data.polyandrous.isInitialized = True
        data.polygynous.isInitialized = True
        data.promiscuous.isInitialized = True
        return data
        # print(parent)
    else:
        return None


def getelementtree(url):
    httpRequest = requests.get(url)
    if httpRequest.status_code != 200:
        return None
    else:
        return BeautifulSoup(httpRequest.text, 'html.parser')
    #     if httpRequest.status_code == 200:
    #         root = ET.fromstring(httpRequest.text)
main()

# def ExtractData():
#     with open(fileName, "r") as species_name:
#         # writer = csv.writer(dataFile, delimiter='\t', quotechar='"', quoting.csv.QUOTE_NONNUMERIC)
#         for line in species_name:
#             name = line.strip().replace(' ', '_')
#             data = DesiredData(False, False, False, False, False,
#              False, False, False, False, False, False)
#              data = ExtractMating(name, data)
#
#
# main()
