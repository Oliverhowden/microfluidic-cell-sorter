from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
t1 = 30
t2 = 70
print(t1, t2)
ffmpeg_extract_subclip("8a.avi", t1, t2, targetname="8aa.avi")
