"""Read the broadcast scoreboard clock (mm:ss) with tesseract."""
import cv2, subprocess, re, numpy as np, tempfile, os
def read_clock(frame):
    crop = frame[38:66, 64:156]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fd, path = tempfile.mkstemp(suffix='.png'); os.close(fd)
    cv2.imwrite(path, b)
    out = subprocess.run(['tesseract', path, '-', '--psm', '7', '-c', 'tessedit_char_whitelist=0123456789:'],
                         capture_output=True, text=True).stdout.strip()
    os.remove(path)
    m = re.fullmatch(r'(\d{2}):?(\d{2})', out.replace(' ', ''))
    return (int(m.group(1)) * 60 + int(m.group(2))) if m else None
if __name__ == '__main__':
    cap = cv2.VideoCapture('clip.mp4'); fps = cap.get(cv2.CAP_PROP_FPS)
    rows = []
    for t in np.arange(5, 362, 5.0):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, f = cap.read()
        if not ok: break
        c = read_clock(f); rows.append((t, c)); print(f'{t:6.1f} {c} {"" if c is None else round(t-c,2)}')
