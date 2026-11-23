"""Download licensed Noto fonts and subset them for the current reading edition.

Run with FontTools and Brotli available. Full upstream files stay in local/;
only the book-specific WOFF2 subsets and original licenses are distributed.
"""
from pathlib import Path
from urllib.request import urlopen, Request
from hashlib import sha256
import json, re, sys, argparse

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--python-libs',type=Path,help='Optional local FontTools installation directory')
args=parser.parse_args()
if args.python_libs: sys.path.insert(0,str(args.python_libs.resolve()))
from fontTools import subset
from fontTools.ttLib import TTFont

CACHE=ROOT/'local/print-fonts'
OUT=ROOT/'assets/fonts'
SOURCES={
 'BookSerif-Regular':('Serif2.003','Serif','Regular','NotoSerifCJKsc'),
 'BookSans-Regular':('Sans2.004','Sans','Regular','NotoSansCJKsc'),
 'BookSans-Bold':('Sans2.004','Sans','Bold','NotoSansCJKsc'),
}

def fetch(url,path):
    if not path.exists():
        with urlopen(Request(url,headers={'User-Agent':'AI-Infra-book-font-build'}),timeout=90) as r:
            data=r.read()
        path.write_bytes(data)
    return path.read_bytes()

def main():
    CACHE.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'README.md',ROOT/'book/前言.md',ROOT/'book/appendices/全书查阅与复算.md']
    groups=json.loads((ROOT/'book/contents.json').read_text(encoding='utf8'))
    for group in groups:
        for n in group['chapters']:
            chapter=ROOT/f'book/chapters/ch{n:02}.md'
            sources.append(chapter)
            images=re.findall(r'^!\[[^\n]*\]\((figures/[^)]+)\)',chapter.read_text(encoding='utf8'),re.M)
            sources.extend(chapter.parent/image for image in images)
    text=''.join(p.read_text(encoding='utf8') for p in sources)+'弦宇由浅入深目录续附录模型表示与学习原理计算架构存储系统算子优化编译分布式训练大模型推理在线服务后训练应用平台工程'
    unicodes=set(map(ord,text))|set(range(32,256))|set(range(0x2000,0x2070))|set(range(0x3000,0x3040))
    entries=[]
    for label,(tag,family,weight,stem) in SOURCES.items():
        url=f'https://raw.githubusercontent.com/notofonts/noto-cjk/{tag}/{family}/OTF/SimplifiedChinese/{stem}-{weight}.otf'
        file=CACHE/f'{stem}-{weight}.otf'; data=fetch(url,file)
        font=TTFont(file,recalcTimestamp=False)
        supported=set(font.getBestCmap())
        # Fail explicitly instead of quietly falling back on another system font.
        han={u for u in unicodes if 0x3400<=u<=0x9fff or 0x20000<=u<=0x323af}
        assert han <= supported, (label, ''.join(map(chr,sorted(han-supported))))
        options=subset.Options(); options.flavor='woff2'; options.layout_features=['*']
        options.name_IDs=['*']; options.name_languages=['*']; options.name_legacy=True
        sub=subset.Subsetter(options=options); sub.populate(unicodes=unicodes&supported); sub.subset(font)
        target=OUT/f'{label}.woff2'; font.flavor='woff2'; font.save(target)
        license_url=f'https://raw.githubusercontent.com/notofonts/noto-cjk/main/{family}/LICENSE'
        fetch(license_url,OUT/f'Noto{family}-LICENSE.txt')
        entries.append(dict(file=target.name,source=url,upstream_sha256=sha256(data).hexdigest(),sha256=sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,cjk_characters=len(han)))
        print(label,target.stat().st_size,flush=True)
    (OUT/'print-fonts.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

if __name__=='__main__': main()
