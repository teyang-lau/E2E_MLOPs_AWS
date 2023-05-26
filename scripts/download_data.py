import urllib
url = 'https://data.gov.sg/api/action/datastore_search?resource_id=f1765b54-a209-4718-8d38-a39237f502b3&limit=5&q=title:jones'
fileobj = urllib.urlopen(url)
print fileobj.read()