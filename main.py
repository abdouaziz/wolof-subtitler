from subtitler import Transcriber


if __name__ == "__main__":

    transcriber = Transcriber()

    transcriber.transcribe(
        "./INTERVIEW-XALAM.mp4",
        output_video_file="output3_video.mp4",
        output_subtitle_file="subtitles.srt",
    )
