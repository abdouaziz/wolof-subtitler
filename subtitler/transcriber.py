from subtitler.video_utils import convert_video_to_audio, create_video_with_subtitles
from subtitler.translator import GoogleTranslator
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

translator = GoogleTranslator()

@dataclass
class TranscriptSegment:
    """Représente un segment de transcription avec des informations sur le locuteur"""
    t0: float  # Heure de début en secondes
    t1: float  # Heure de fin en secondes
    text: str  # Texte transcrit
    speaker: str  # Identifiant du locuteur



class Transcriber:
    """
    Une classe pour gérer la transcription vidéo et la génération de sous-titres avec diarisation des locuteurs.
    """
    
    def __init__(self):
        """
        Initialise le Transcriber avec un modèle Whisper spécifié et un pipeline de diarisation des locuteurs.
        
        Args:
            model_name (str): Nom/chemin du modèle Whisper à utiliser
        """
        self.model = WhisperTranscriber()
        
        # Initialise le pipeline de diarisation des locuteurs
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="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"
        )
        self.diarization_pipeline.to(torch.device("cuda"))
        
    def _split_audio(self, audio_path):
        """
        Divise le fichier audio en segments plus petits pour une meilleure précision de la transcription en utilisant pyannote.audio.
        
        Args:
            audio_path (str): Chemin du fichier audio
            
        Returns:
            list[tuple]: Liste de tuples (heure_début, heure_fin, chemin_segment)
        """
        audio = AudioSegment.from_wav(audio_path)
        duration_ms = len(audio)
        
        segments = []
        for turn, _, speaker in self.diarization_pipeline(audio_path).itertracks(yield_label=True):
            start_time = turn.start
            end_time = turn.end
            
            # Créer un fichier temporaire pour le segment
            temp_segment_path = f"temp_segment_{start_time:.2f}_{end_time:.2f}.wav"
            audio[int(start_time * 1000):int(end_time * 1000)].export(temp_segment_path, format="wav")
            
            segments.append((
                start_time,
                end_time,
                temp_segment_path
            ))
            
        return segments

    def _perform_speaker_diarization(self, audio_path):
        """
        Effectue la diarisation des locuteurs sur le fichier audio.
        
        Args:
            audio_path (str): Chemin du fichier audio
            
        Returns:
            dict: Mappage des identifiants de locuteur à leurs segments de parole correspondants
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
        Transcrit un fichier vidéo avec diarisation des locuteurs et crée éventuellement une vidéo avec sous-titres.
        
        Args:
            video_file (str): Chemin du fichier vidéo d'entrée
            output_video_file (str, optional): Chemin du fichier vidéo de sortie avec sous-titres
            output_subtitle_file (str): Chemin du fichier de sous-titres de sortie
            
        Raises:
            Exception: En cas d'échec de la transcription ou de la génération des sous-titres
        """
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Fichier vidéo introuvable : {video_file}")

        transcript = []
        temp_audio = "temporary_audio.wav"
        temp_segments = []
        
        try:
            logger.info("Conversion de la vidéo en audio...")
            convert_video_to_audio(video_file, temp_audio)
            logger.info("Conversion audio terminée.")

            logger.info("Division de l'audio en segments...")
            audio_segments = self._split_audio(temp_audio)
            
            logger.info("Effectuer la diarisation des locuteurs...")
            speaker_segments = self._perform_speaker_diarization(temp_audio)
            
            logger.info("Transcription des segments audio...")
            for start_time, end_time, segment_path in audio_segments:
                temp_segments.append(segment_path)
                
                # Transcrire le segment
                raw_segment_transcript = self.model(segment_path)
                
                # Traiter la transcription du segment et ajuster les horodatages
                segment_transcript = self._process_raw_transcript(
                    raw_segment_transcript,
                    base_time=start_time,
                    speaker_segments=speaker_segments
                )
                
                transcript.extend(segment_transcript)
            
            logger.info("Transcription terminée.")
            
        except Exception as e:
            logger.error(f"Erreur lors de la transcription : {str(e)}")
            raise
        finally:
            # Nettoyer les fichiers temporaires
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            for segment_path in temp_segments:
                if os.path.exists(segment_path):
                    os.remove(segment_path)

        # Garantir les extensions de fichiers appropriées
        if not output_subtitle_file.endswith(".srt"):
            output_subtitle_file += ".srt"
            
        if output_video_file and not any(output_video_file.endswith(ext) for ext in ['.mp4', '.avi', '.mkv']):
            output_video_file += '.mp4'

        try:
            # Créer un fichier de contrôle sans traduction
            checkpoint_path = f"checkpoint_{output_subtitle_file}"
            checkpoint_content = self.generate_srt_file(
                transcript,
                with_translation=False,
                suppress_output=True
            )
            
            self._write_srt_file(checkpoint_path, checkpoint_content)
            
            # Générer le fichier SRT final avec traduction
            logger.info("Génération du fichier SRT...")
            final_content = self.generate_srt_file(
                transcript,
                with_translation=True,
                suppress_output=False
            )
            
            self._write_srt_file(output_subtitle_file, final_content)
            logger.info(f"Fichier SRT généré : {output_subtitle_file}")

            # Créer une vidéo avec sous-titres si demandé
            if output_video_file:
                logger.info("Création de la vidéo avec sous-titres...")
                create_video_with_subtitles(
                    video_file,
                    output_subtitle_file,
                    output_video_file
                )
                logger.info("Vidéo avec sous-titres créée avec succès.")

        except Exception as e:
            logger.warning(f"Erreur lors de la génération des sous-titres ou de la création de la vidéo : {str(e)}")
            raise

        return output_subtitle_file
    
    def _process_raw_transcript(self, raw_transcript, base_time=0.0, speaker_segments=None):
        """
        Convertit les données de transcription brutes en objets TranscriptSegment avec les horodatages et les informations sur le locuteur ajustés.
        
        Args:
            raw_transcript: Données de transcription brutes du modèle
            base_time (float): Décalage horaire de base pour le segment
            speaker_segments (dict): Mappage des identifiants de locuteur à leurs segments de parole
            
        Returns:
            list[TranscriptSegment]: Liste des segments de transcription traités
        """
        segments = []
        
        if speaker_segments:
            # Attribue les identifiants de locuteur aux segments de transcription en fonction de la diarisation
            speaker_map = {}
            for speaker, speaker_times in speaker_segments.items():
                for start_time, end_time in speaker_times:
                    speaker_map[(start_time, end_time)] = speaker
        
        if isinstance(raw_transcript, str):
            # Si c'est juste une chaîne, créez un seul segment avec un minutage estimé
            segments.append(TranscriptSegment(
                t0=base_time,
                t1=base_time + 2,
                text=raw_transcript,
                speaker=""
            ))
        elif isinstance(raw_transcript, list):
            for i, segment in enumerate(raw_transcript):
                if isinstance(segment, dict):
                    # Gérer le format de dictionnaire avec ajustement des horodatages
                    start_time = base_time + float(segment.get('start', i * 1.0))
                    end_time = base_time + float(segment.get('end', (i + 1) * 1.0))
                    
                    # Récupérer l'identifiant du locuteur à partir de la diarisation, si disponible
                    speaker = speaker_map.get((start_time, end_time), f"")
                    
                    segments.append(TranscriptSegment(
                        t0=start_time,
                        t1=end_time,
                        text=segment.get('text', ''),
                        speaker=speaker
                    ))
                elif isinstance(segment, str):
                    # Gérer le format de chaîne avec un minutage estimé
                    segments.append(TranscriptSegment(
                        t0=base_time + i * 2.0,
                        t1=base_time + (i + 1) * 2.0,
                        text=segment,
                        speaker=f""
                    ))
                else:
                    # Si le segment a déjà le bon format, utilisez-le directement
                    segments.append(TranscriptSegment(
                        t0=base_time + segment.t0,
                        t1=base_time + segment.t1,
                        text=segment.text,
                        speaker=segment.speaker
                    ))
        
        return segments

    def generate_srt_file(self, transcript, with_translation=False, suppress_output=False):
        """
        Génère le contenu SRT à partir de la transcription avec traduction optionnelle et informations sur le locuteur.
        
        Args:
            transcript: Liste des segments de transcription
            with_translation (bool): Si la traduction doit être incluse
            suppress_output (bool): Si les sorties de progression doivent être supprimées
            
        Returns:
            str: Contenu SRT généré
        """
        if not transcript:
            return ""

        # Trier les segments par heure de début pour assurer un bon ordre
        transcript.sort(key=lambda x: x.t0)
        
        # Fusionner les segments très proches ou se chevauchant
        merged_transcript = self._merge_close_segments(transcript)
        
        srt_content = []
        
        if not suppress_output:
            print(f"Total lines: {len(merged_transcript)}")

        for count, line in enumerate(merged_transcript, 1):
            timestamp = (
                f"{self.format_seconds_to_srt_timestamp(line.t0)} --> "
                f"{self.format_seconds_to_srt_timestamp(line.t1)}"
            )
            
            text = f"{line.text.strip()} {line.speaker}"
            if with_translation:
                translated_text = translator(text=line.text).strip()
                
                if not suppress_output:
                    print(f"- Line {count} of {len(merged_transcript)}: {line.text} - {line.speaker}\n --> {translated_text} - {line.speaker}")
                text = f"{translated_text} {line.speaker}"
            elif not suppress_output:
                print(f"- Line {count} of {len(merged_transcript)}: {line.text} - {line.speaker}")

            srt_content.extend([
                str(count),
                timestamp,
                text,
                ""
            ])

        return "\n".join(srt_content)

    def _merge_close_segments(self, transcript, time_threshold=0.3):
        """
        Fusionne les segments très proches ou se chevauchant.
        
        Args:
            transcript (list[TranscriptSegment]): Liste des segments de transcription
            time_threshold (float): Écart de temps maximum entre les segments à fusionner
            
        Returns:
            list[TranscriptSegment]: Liste des segments fusionnés
        """
        if not transcript:
            return []
            
        merged = []
        current = transcript[0]
        
        for next_segment in transcript[1:]:
            if next_segment.t0 - current.t1 <= time_threshold:
                # Fusion des segments
                current = TranscriptSegment(
                    t0=current.t0,
                    t1=next_segment.t1,
                    text=f"{current.text} {next_segment.text}",
                    speaker=current.speaker
                )
            else:
                merged.append(current)
                current = next_segment
                
        merged.append(current)
        return merged

    @staticmethod
    def _write_srt_file(filepath, content):
        """
        Écrit le contenu SRT dans un fichier avec l'encodage approprié.
        """
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            raise IOError(f"Échec de l'écriture du fichier SRT {filepath} : {str(e)}")

    @staticmethod
    def format_seconds_to_srt_timestamp(seconds):
        """
        Convertit les secondes au format d'horodatage SRT.
        """
        milliseconds = round(seconds * 1000)
        
        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000
        
        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000
        
        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"