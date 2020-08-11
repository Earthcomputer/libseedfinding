#!/usr/bin/env python3
import json
import urllib.request
from urllib.error import HTTPError, URLError
import fileinput
import os
def download_file(url, filename):
    try:
        print(f'Downloading {filename}...')
        f = urllib.request.urlopen(url)
        with open(filename, 'wb+') as local_file:
            local_file.write(f.read())
    except HTTPError as e:
        print('HTTP Error')
        print(e)
    except URLError as e:
        print('URL Error')
        print(e)

download_file("https://launchermeta.mojang.com/mc/game/version_manifest.json", f"version_manifest.json")
text=[]
with open("version_manifest.json") as file:
    versions=json.load(file)["versions"]
    versions.reverse()
    typf=lambda typ:"SNAPSHOT" if typ=="snapshot" else "RELEASE" if typ=="release" else "OLD_ALPHA" if typ=="old_alpha" else "OLD_BETA" if typ=="old_beta" else "NONE"
    for i,version in enumerate(versions):
        text.append("const Version MC_"+version.get("id").replace(".","_").replace("-","_").replace(" ","_")+" = {"+str(i)+","+typf(version.get("type"))+',"'+version.get("releaseTime")+'"};')



with fileinput.FileInput("../src/version.h", inplace=True, backup='.bak') as file:
    for line in file:
        print(line.replace("/////////////////////////GENERATION////////////////////////////////", (os.linesep+"\t").join(text)), end='')
