from subtitler.video_utils import convert_video_to_audio, create_video_with_subtitles
from subtitler.translator import Translator
from subtitler.utils import WhisperTranscriber
from subtitler.log import logger
import os
import warnings
from dataclasses import dataclass
import numpy as np
from pydub import AudioSegment

warnings.filterwarnings("ignore")

translator = Translator()

@dataclass
class TranscriptSegment:
    """Represents a single segment of transcription with timing information"""
    t0: float  # Start time in seconds
    t1: float  # End time in seconds
    text: str  # Transcribed text

class Transcriber:
    """
    A class to handle video transcription and subtitle generation with improved timestep handling.
    """
    
    def __init__(self, model_name="LiquAId/whisper-tiny-french-HanNeurAI", segment_duration=30):
        """
        Initialize the Transcriber with a specified whisper model.
        
        Args:
            model_name (str): Name/path of the whisper model to use
            segment_duration (int): Duration of each audio segment in seconds
        """
        self.model = WhisperTranscriber(model_name)
        self.segment_duration = segment_duration
        
    def _split_audio(self, audio_path):
        """
        Split audio file into smaller segments for better transcription accuracy.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            list[tuple]: List of (start_time, end_time, audio_segment) tuples
        """
        audio = AudioSegment.from_wav(audio_path)
        duration_ms = len(audio)
        segment_duration_ms = self.segment_duration * 1000
        
        segments = []
        for start_ms in range(0, duration_ms, segment_duration_ms):
            end_ms = min(start_ms + segment_duration_ms, duration_ms)
            segment = audio[start_ms:end_ms]
            
            # Create temporary file for segment
            temp_segment_path = f"temp_segment_{start_ms}.wav"
            segment.export(temp_segment_path, format="wav")
            
            segments.append((
                start_ms / 1000,  # Convert to seconds
                end_ms / 1000,
                temp_segment_path
            ))
            
        return segments

    def transcribe(self, video_file, output_video_file=None, output_subtitle_file="output.srt"):
        """
        Transcribe a video file with improved timestep handling and optionally create subtitled video.
        
        Args:
            video_file (str): Path to input video file
            output_video_file (str, optional): Path for output video with subtitles
            output_subtitle_file (str): Path for output SRT file
            
        Raises:
            Exception: If transcription or subtitle generation fails
        """
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")

        transcript = []
        temp_audio = "temporary_audio.wav"
        temp_segments = []
        
        try:
            logger.info("Converting video to audio...")
            convert_video_to_audio(video_file, temp_audio)
            logger.info("Audio conversion completed.")

            logger.info("Splitting audio into segments...")
            audio_segments = self._split_audio(temp_audio)
            
            logger.info("Transcribing audio segments...")
            for start_time, end_time, segment_path in audio_segments:
                temp_segments.append(segment_path)
                
                # Transcribe segment
                raw_segment_transcript = self.model(segment_path)
                
                # Process segment transcript and adjust timestamps
                segment_transcript = self._process_raw_transcript(
                    raw_segment_transcript,
                    base_time=start_time
                )
                
                transcript.extend(segment_transcript)
            
            logger.info("Transcription completed.")
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise
        finally:
            # Clean up temporary files
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            for segment_path in temp_segments:
                if os.path.exists(segment_path):
                    os.remove(segment_path)

        # Ensure proper file extensions
        if not output_subtitle_file.endswith(".srt"):
            output_subtitle_file += ".srt"
            
        if output_video_file and not any(output_video_file.endswith(ext) for ext in ['.mp4', '.avi', '.mkv']):
            output_video_file += '.mp4'

        try:
            # Create checkpoint file without translation
            checkpoint_path = f"checkpoint_{output_subtitle_file}"
            checkpoint_content = self.generate_srt_file(
                transcript,
                with_translation=False,
                suppress_output=True
            )
            
            self._write_srt_file(checkpoint_path, checkpoint_content)
            
            # Generate final SRT with translation
            logger.info("Generating SRT file...")
            final_content = self.generate_srt_file(
                transcript,
                with_translation=True,
                suppress_output=False
            )
            
            self._write_srt_file(output_subtitle_file, final_content)
            logger.info(f"SRT file generated: {output_subtitle_file}")

            # Create video with subtitles if requested
            if output_video_file:
                logger.info("Creating video with subtitles...")
                create_video_with_subtitles(
                    video_file,
                    output_subtitle_file,
                    output_video_file
                )
                logger.info("Video with subtitles created successfully.")

        except Exception as e:
            logger.warning(f"Error during subtitle generation or video creation: {str(e)}")
            raise

        return output_subtitle_file
    
    def _process_raw_transcript(self, raw_transcript, base_time=0.0):
        """
        Convert raw transcript data into TranscriptSegment objects with adjusted timestamps.
        
        Args:
            raw_transcript: Raw transcription data from the model
            base_time (float): Base time offset for the segment
            
        Returns:
            list[TranscriptSegment]: List of processed transcript segments
        """
        segments = []
        
        if isinstance(raw_transcript, str):
            # If it's just a string, create segments with smaller time intervals
            words = raw_transcript.split()
            words_per_segment = 3  # Adjust this value as needed
            
            for i in range(0, len(words), words_per_segment):
                segment_words = words[i:i + words_per_segment]
                segment_text = " ".join(segment_words)
                
                # Estimate 0.3 seconds per word
                t0 = base_time + (i / words_per_segment) * (words_per_segment * 0.3)
                t1 = t0 + len(segment_words) * 0.3
                
                segments.append(TranscriptSegment(t0=t0, t1=t1, text=segment_text))
                
        elif isinstance(raw_transcript, list):
            for i, segment in enumerate(raw_transcript):
                if isinstance(segment, dict):
                    # Handle dictionary format with adjusted timestamps
                    segments.append(TranscriptSegment(
                        t0=base_time + float(segment.get('start', i * 2.0)),
                        t1=base_time + float(segment.get('end', (i + 1) * 2.0)),
                        text=segment.get('text', '')
                    ))
                elif isinstance(segment, str):
                    # Handle string format with estimated timing
                    segments.append(TranscriptSegment(
                        t0=base_time + i * 2.0,
                        t1=base_time + (i + 1) * 2.0,
                        text=segment
                    ))
                else:
                    # Adjust timestamps of existing segments
                    segments.append(TranscriptSegment(
                        t0=base_time + segment.t0,
                        t1=base_time + segment.t1,
                        text=segment.text
                    ))
        
        return segments

    def generate_srt_file(self, transcript, with_translation=False, suppress_output=False):
        """
        Generate SRT content from transcript with optional translation.
        
        Args:
            transcript: List of transcription segments
            with_translation (bool): Whether to translate the text
            suppress_output (bool): Whether to suppress progress output
            
        Returns:
            str: Generated SRT content
        """
        if not transcript:
            return ""

        # Sort segments by start time to ensure proper ordering
        transcript.sort(key=lambda x: x.t0)
        
        # Merge very close or overlapping segments
        merged_transcript = self._merge_close_segments(transcript)
        
        srt_content = []
        
        if not suppress_output:
            print(f"Total lines: {len(merged_transcript)}")

        for count, line in enumerate(merged_transcript, 1):
            timestamp = (
                f"{self.format_seconds_to_srt_timestamp(line.t0)} --> "
                f"{self.format_seconds_to_srt_timestamp(line.t1)}"
            )
            
            text = line.text.strip()
            if with_translation:
                translated_text = translator.predict(text=text).strip()
                
                if not suppress_output:
                    print(f"- Line {count} of {len(merged_transcript)}: {text}\n --> {translated_text}")
                text = translated_text
            elif not suppress_output:
                print(f"- Line {count} of {len(merged_transcript)}: {text}")

            srt_content.extend([
                str(count),
                timestamp,
                text,
                ""
            ])

        return "\n".join(srt_content)

    def _merge_close_segments(self, transcript, time_threshold=0.3):
        """
        Merge segments that are very close in time or overlapping.
        
        Args:
            transcript (list[TranscriptSegment]): List of transcript segments
            time_threshold (float): Maximum time gap between segments to merge
            
        Returns:
            list[TranscriptSegment]: List of merged segments
        """
        if not transcript:
            return []
            
        merged = []
        current = transcript[0]
        
        for next_segment in transcript[1:]:
            if next_segment.t0 - current.t1 <= time_threshold:
                # Merge segments
                current = TranscriptSegment(
                    t0=current.t0,
                    t1=next_segment.t1,
                    text=f"{current.text} {next_segment.text}"
                )
            else:
                merged.append(current)
                current = next_segment
                
        merged.append(current)
        return merged

    @staticmethod
    def _write_srt_file(filepath, content):
        """Write SRT content to file with proper encoding."""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            raise IOError(f"Failed to write SRT file {filepath}: {str(e)}")

    @staticmethod
    def format_seconds_to_srt_timestamp(seconds):
        """Convert seconds to SRT timestamp format."""
        milliseconds = round(seconds * 1000)
        
        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000
        
        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000
        
        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"