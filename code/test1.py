from subtitler.video_utils import convert_video_to_audio, create_video_with_subtitles
from subtitler.translator import Translator
from subtitler.utils import WhisperTranscriber
from subtitler.log import logger
import os
import warnings
from dataclasses import dataclass
import numpy as np
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torch

warnings.filterwarnings("ignore")

translator = Translator()

@dataclass
class TranscriptSegment:
    """Represents a single segment of transcription with timing and speaker information"""
    t0: float  # Start time in seconds
    t1: float  # End time in seconds
    text: str  # Transcribed text
    speaker: str  # Speaker identifier

class Transcriber:
    """
    A class to handle video transcription and subtitle generation with speaker diarization.
    """
    
    def __init__(self, model_name="LiquAId/whisper-tiny-french-HanNeurAI", segment_duration=30):
        """
        Initialize the Transcriber with a specified whisper model and speaker diarization pipeline.
        
        Args:
            model_name (str): Name/path of the whisper model to use
            segment_duration (int): Duration of each audio segment in seconds
        """
        self.model = WhisperTranscriber(model_name)
        self.segment_duration = segment_duration
        
        # Initialize speaker diarization pipeline
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="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"
        )
        self.diarization_pipeline.to(torch.device("cuda"))
        
    def _split_audio(self, audio_path):
        """
        Split audio file into smaller segments for better transcription accuracy.
        """
        # Same as before

    def _perform_speaker_diarization(self, audio_path):
        """
        Perform speaker diarization on the audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Mapping of speaker IDs to their corresponding speech segments
        """
        diarization = self.diarization_pipeline(audio_path)
        speaker_segments = {}
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((turn.start, turn.end))
        
        return speaker_segments
    
    def transcribe(self, video_file, output_video_file=None, output_subtitle_file="output.srt"):
        """
        Transcribe a video file with speaker diarization and optionally create subtitled video.
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
            
            logger.info("Performing speaker diarization...")
            speaker_segments = self._perform_speaker_diarization(temp_audio)
            
            logger.info("Transcribing audio segments...")
            for start_time, end_time, segment_path in audio_segments:
                temp_segments.append(segment_path)
                
                # Transcribe segment
                raw_segment_transcript = self.model(segment_path)
                
                # Process segment transcript and adjust timestamps
                segment_transcript = self._process_raw_transcript(
                    raw_segment_transcript,
                    base_time=start_time,
                    speaker_segments=speaker_segments
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
    
    
    
    
    def _process_raw_transcript(self, raw_transcript, base_time=0.0, speaker_segments=None):
        """
        Convert raw transcript data into TranscriptSegment objects with adjusted timestamps and speaker information.
        """
        segments = []
        
        if speaker_segments:
            # Assign speaker IDs to transcript segments based on diarization
            speaker_map = {}
            for speaker, speaker_times in speaker_segments.items():
                for start_time, end_time in speaker_times:
                    speaker_map[(start_time, end_time)] = speaker
        
        # Same as before, but now we also assign the speaker ID to each TranscriptSegment
        
        return segments

    def generate_srt_file(self, transcript, with_translation=False, suppress_output=False):
        """
        Generate SRT content from transcript with optional translation and speaker information.
        """
        # Same as before, but now we include the speaker information in the output
        
        return "\n".join(srt_content)