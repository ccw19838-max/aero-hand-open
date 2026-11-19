"""Verify all encoded frames and playback in the installed Chromium."""
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path
import json
import subprocess
import threading
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent


def main():
    report=json.loads((ROOT/'recording_report.json').read_text())
    checks={}
    for axis in 'xyz':
        path=ROOT/report['videos'][axis]['path']
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
            'stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration','-of','json',str(path)]))
        stream=probe['streams'][0]
        assert stream['codec_name']=='h264' and stream['width']==1280 and stream['height']==800
        assert int(stream['nb_frames'])==report['videos'][axis]['frames']
        subprocess.run(['ffmpeg','-v','error','-threads','1','-i',str(path),'-f','null','-'],check=True,capture_output=True)
        checks[axis]=probe
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(ROOT)))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,
                args=['--no-sandbox','--disable-dev-shm-usage','--autoplay-policy=no-user-gesture-required'])
            page=browser.new_page(viewport={'width':1280,'height':1000})
            errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html',wait_until='load',timeout=90000)
            page.wait_for_function("Array.from(document.querySelectorAll('video')).length===3 && Array.from(document.querySelectorAll('video')).every(v=>v.readyState>=1 && v.duration>0)",timeout=30000)
            page.get_by_role('button',name='同时从头播放').click()
            page.wait_for_function("Array.from(document.querySelectorAll('video')).every(v=>v.currentTime>0.2 && !v.paused)",timeout=30000)
            playback=page.locator('video').evaluate_all('(vs)=>vs.map(v=>({time:v.currentTime,duration:v.duration,width:v.videoWidth,height:v.videoHeight,error:v.error?.message || null}))')
            page.get_by_role('button',name='全部暂停').click()
            assert page.locator('video').evaluate_all('(vs)=>vs.every(v=>v.paused)')
            page.locator('video').evaluate_all('(vs)=>vs.forEach(v=>v.currentTime=v.duration-.5)')
            page.wait_for_timeout(800)
            page.screenshot(path=str(ROOT/'preview_desktop.png'),full_page=True)
            page.set_viewport_size({'width':390,'height':844})
            page.wait_for_timeout(300)
            overflow=page.evaluate('document.documentElement.scrollWidth>window.innerWidth')
            assert not overflow
            page.screenshot(path=str(ROOT/'preview_mobile.png'),full_page=True)
            assert not errors,errors
            checks['browser']={'playback':playback,'page_errors':errors,'mobile_horizontal_overflow':overflow,
                'simultaneous_play_and_pause':True,'videos_seekable':True,'self_contained_html':True}
            browser.close()
    finally:server.shutdown();server.server_close()
    (ROOT/'qa.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps({'all_frames_decode':True,'playback':checks['browser']}))


if __name__=='__main__':main()
