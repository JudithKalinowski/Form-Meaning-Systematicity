# -*- coding: utf-8 -*-
"""

In this py.-file we get IPA-transcriptions from the website NOAB by web scraping

Created on Tue Dec 14 13:29:48 2021
@author: Judith Kalinowski

"""

import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
import time

NOR_items = pd.read_csv("C:/Users/judit/OneDrive/Desktop/Vocab_Prediction/Prediction/NOR_items.csv", header=int(),
                        index_col=0, encoding='ANSI')

items_with_id = NOR_items[['definition', 'num_item_id']]
NOR_items.insert(3, "IPA", "")
NOR_items.insert(2, "definition_new", NOR_items['definition'])

for word in NOR_items["definition_new"]:
    if "(" in word:
        if word[0] == "(":
            word_new = word.split(")",maxsplit=1)[-1]
        else:
            word_new = word.split("(")[0]
        NOR_items.at[NOR_items[NOR_items['definition_new'] == word].index.values[0], 'definition_new'] = word_new
    else:
        word_new = word
        
NOR_items_IPA = NOR_items.iloc[:731, :]
items_list = NOR_items_IPA.iloc[:, 2].tolist()

print(NOR_items_IPA['definition_new'])

# Instantiate options
opts = Options()

# opts.add_argument(" — headless") # Uncomment if the headless version needed
#opts.binary_location = "C:/ProgramData/Microsoft/Windows/Start Menu/Programs"

# Set the location of the webdriver
firefox_driver = "C:/Users/judit/geckodriver.exe"

# Instantiate a webdriver
driver = webdriver.Firefox(options=opts, executable_path=firefox_driver)

time.sleep(1)

# Load the HTML page
for name in items_list:
    print(name)
    driver.get("https://naob.no/ordbok/" + name)
    time.sleep(1)
    
    # Parse processed webpage with BeautifulSoup
    soup = BeautifulSoup(driver.page_source)
    soup_str = '%s'%soup
    #print(soup.find("ekorn").get_text())
    #if name in str(soup):
        #print(str(soup)) etter, og
    if "UTTALE" in soup_str:
        if (len(soup_str.split("UTTALE</span><span>[", maxsplit=1)[-1].split(']</span>')[0])) > 50:
            IPA = None
        else:
            IPA = soup_str.split("UTTALE</span><span>[", maxsplit=1)[-1].split(']</span>')[0]
        NOR_items_IPA.at[NOR_items[NOR_items['definition_new'] == name].index.values[0], 'IPA'] = IPA
    else:
        NOR_items_IPA.at[NOR_items[NOR_items['definition_new'] == name].index.values[0], 'IPA'] = None

save_to = 'C:/Users/judit/OneDrive/Desktop/Vocab_Prediction/csv/'
NOR_items_IPA.to_csv(save_to + 'NOR_items_with_IPA.csv')

IPA_list = NOR_items_IPA['IPA'].tolist()
None_items = sum(x is None for x in IPA_list)
All_items = len(IPA_list)
print(None_items/All_items)