#sys.path.append('./platform')
import video_tools
video_recorder = video_tools.VideoRecorder(0)


video_recorder.record_to_file(video_filepath,  recording_time)

video_recorder.wait_until_idle()
