import requests
from bs4 import BeautifulSoup
import os


params = (
    ('_ajax', '1'),
)

headers = {
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Referer': 'https://artuk.org/discover/artworks/search/class_title:landscape--category:countryside/page/0',
    'X-Requested-With': 'XMLHttpRequest',
    'Connection': 'keep-alive',
}

artuk_url = "https://artuk.org/discover/artworks/search/class_title:landscape--category:countryside/page/"

def get_html_to_parse(page):
    response = requests.get(
        f'{artuk_url}{page}', params=params, headers=headers)
    return response.json()


def json_to_bs4(json):
    return BeautifulSoup(json.get('html', ''), 'lxml')


def get_discovers(bs4_obj):
    links = bs4_obj.find_all('a')
    relevant_links = []
    for link in links:
        if "discover" in link['href'].split("/")[3]:
            relevant_links.append(link['href'])
    return relevant_links


def save_img_from_url(img_url, name):

    bs4_html = html_to_bs4(requests.get(img_url).text)
    divs = bs4_html.find('div', {'class': 'artwork'})
    img_src = divs.find('img')['src']

    response = requests.get(img_src)
    if response.status_code == 200:
        if not os.path.exists("output"):
            os.makedirs("output")
        with open(f"output/{name}.png", 'wb+') as f:
            f.write(response.content)


def html_to_bs4(html):
    return BeautifulSoup(html, 'lxml')


if __name__ == "__main__":
    counter = 0
    for i in range(0, 1000):
        try:
            html_to_parse = get_html_to_parse(i)
            bs4_obj = json_to_bs4(html_to_parse)
            relevant = get_discovers(bs4_obj)
            for link in relevant:
                print(link)
                try:
                    save_img_from_url(link, counter)
                    counter += 1
                except:
                    print("Error...")
                    continue
        except:
            continue




