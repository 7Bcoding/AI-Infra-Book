"""Build a static GitHub Pages reader from the reviewed offline edition."""
from pathlib import Path
import argparse, html, re, shutil

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',default='7Bcoding/AI-Infra-Book')
    p.add_argument('--output',type=Path,default=ROOT/'_site')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',a.repo):
        p.error('Expected owner/repository')
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    downloads=out/'downloads';downloads.mkdir(exist_ok=True)
    for source,target in [('AI-Infra-全书.html','book.html'),('AI-Infra-由浅入深-CQY.pdf','book.pdf'),('AI-Infra-六篇实验.zip','examples.zip')]:
        shutil.copyfile(ROOT/'dist'/source,downloads/target)
    links=[]
    for n in range(1,26):
        title=(ROOT/f'book/chapters/ch{n:02}.md').read_text(encoding='utf8').splitlines()[0].lstrip('# ')
        links.append(f'<li><a href="downloads/book.html#ch{n:02}">{html.escape(title)}</a></li>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Infra 由浅入深 · CQY</title><style>body{max-width:760px;margin:60px auto;padding:0 24px;color:#25313a;background:#fff;font:18px/1.9 system-ui,sans-serif}h1{font-size:38px;line-height:1.4}a{color:#285a70;text-underline-offset:4px}nav{display:flex;gap:20px;flex-wrap:wrap;margin:28px 0}li{margin:9px 0}footer{border-top:1px solid #ddd;margin-top:40px;font-size:14px}@media(max-width:600px){body{margin:28px auto;font-size:16px}h1{font-size:30px}}</style><main><h1>AI Infra 由浅入深</h1><p>CQY 著</p><p>从数学模型与学习原理出发，逐步理解计算架构、算子与编译、分布式训练、推理服务和平台工程。用推导、可复算的例子和图解，讨论每一种设计解决的问题与付出的代价。</p><nav><a href="downloads/book.html#foreword">在线阅读</a><a href="downloads/book.pdf">下载 PDF</a><a href="downloads/book.html" download>离线 HTML</a><a href="downloads/examples.zip">Markdown 合订本与实验</a><a href="https://github.com/REPO">Markdown 与仓库</a></nav><p>六篇 · 二十五章 · FIGURE_COUNT 幅图解 · EXERCISE_COUNT 道附解答练习</p><h2>全书目录</h2><ol>CHAPTERS</ol><footer><p>© 2026 CQY。保留版权，免费阅读和下载。<a href="LICENSE.txt">使用说明</a> · <a href="THIRD_PARTY_NOTICES.txt">第三方说明</a> · <a href="https://github.com/REPO/issues">勘误反馈</a></p></footer></main></html>'''
    figure_index = (ROOT/'docs/全书图解索引.md').read_text(encoding='utf8')
    exercise_index = (ROOT/'docs/练习答案核对.md').read_text(encoding='utf8')
    figure_count = figure_index.count('\n- [图 ')
    exercises = re.findall(r'\| (\d+) \| \[第 \d+ 章练习', exercise_index)
    if not figure_count or len(exercises) != len(links):
        raise ValueError('Reading indexes do not match the chapter list')
    page = page.replace('FIGURE_COUNT', str(figure_count))
    page = page.replace('EXERCISE_COUNT', str(sum(map(int, exercises))))
    (out/'index.html').write_text(page.replace('REPO', a.repo).replace('CHAPTERS', ''.join(links)), encoding='utf8')
    (out/'.nojekyll').touch()
    for source,target in [('LICENSE','LICENSE.txt'),('THIRD_PARTY_NOTICES.md','THIRD_PARTY_NOTICES.txt')]:
        if (ROOT/source).exists():shutil.copyfile(ROOT/source,out/target)
    print(f'Site generated: {out.resolve()}')

if __name__=='__main__':main()
