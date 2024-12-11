"""
In this script we will parse the FAQ data and convert it into a format that can be used by the RAG model.
url for parsing is https://www.vsb.cz/en/study/degree-students/faqs/
"""

# get html content inside class .content-container

import requests
from bs4 import BeautifulSoup

url = "https://www.vsb.cz/en/study/degree-students/faqs/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
content = soup.find_all(class_="content-container")
print(content)

# dump content to file
with open("data/faq.html", "w") as file:
    file.write(str(content))
