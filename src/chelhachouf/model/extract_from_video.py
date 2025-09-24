import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Union

class QwenVideoTextReader:
    def __init__(self, model_name="/home/afafelwafi/ocr_amazigh/amazigh_ocr_model/checkpoint-500", device="auto"):
        """
        Initialize Qwen2.5-VL model for video text reading
        
        Args:
            model_name: HuggingFace model name
            device: Device to run model on ("auto", "cuda", "cpu")
        """
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        print(f"Loading model {model_name} on {self.device}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
            device_map="auto" if self.device == "cuda" else None,
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if self.device == "cpu":
            self.model = self.model.to("cpu")
    
    def extract_frames(self, video_path: str, max_frames: int = 8, method: str = "uniform", 
                      interval_seconds: int = 30, time_range: tuple = None) -> List[tuple]:
        """
        Extract frames from video for text detection (optimized for long videos)
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            method: Frame extraction method ("uniform", "interval", "adaptive", "segments")
            interval_seconds: Seconds between frames for interval method
            time_range: (start_seconds, end_seconds) to process only part of video
        
        Returns:
            List of tuples (PIL Image, timestamp_seconds)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_seconds = total_frames / fps
        
        print(f"Video info: {duration_seconds/60:.1f} minutes, {fps:.1f} FPS, {total_frames} total frames")
        
        # Handle time range
        start_frame = 60
        end_frame = total_frames - 1
        if time_range:
            start_frame = max(0, int(time_range[0] * fps))
            end_frame = min(total_frames - 1, int(time_range[1] * fps))
            print(f"Processing time range: {time_range[0]/60:.1f} - {time_range[1]/60:.1f} minutes")
        
        if method == "uniform":
            # Extract frames uniformly across video/time range
            frame_indices = np.linspace(start_frame, end_frame, max_frames, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    timestamp = frame_idx / fps
                    frames.append((pil_image, timestamp))
        
        elif method == "interval":
            # Extract frames at regular time intervals
            current_time = start_frame / fps
            end_time = end_frame / fps
            
            while current_time <= end_time and len(frames) < max_frames:
                frame_idx = int(current_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append((pil_image, current_time))
                
                current_time += interval_seconds
        
        elif method == "segments":
            # Divide video into segments and take one frame from each
            segment_duration = (end_frame - start_frame) / fps / max_frames
            
            for i in range(max_frames):
                segment_start = start_frame + i * segment_duration * fps
                # Take frame from middle of segment
                frame_idx = int(segment_start + (segment_duration * fps) / 2)
                
                if frame_idx <= end_frame:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                        timestamp = frame_idx / fps
                        frames.append((pil_image, timestamp))
        
        elif method == "adaptive":
            # For long videos: sample more at beginning/end, less in middle
            total_duration = (end_frame - start_frame) / fps
            
            # More samples at start (first 10% of video)
            start_samples = max(2, max_frames // 4)
            start_duration = total_duration * 0.1
            
            # More samples at end (last 10% of video)
            end_samples = max(2, max_frames // 4)
            end_duration = total_duration * 0.1
            
            # Remaining samples in middle
            middle_samples = max_frames - start_samples - end_samples
            
            # Start samples
            for i in range(start_samples):
                timestamp = (start_frame / fps) + (i * start_duration / start_samples)
                frame_idx = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append((pil_image, timestamp))
            
            # Middle samples
            middle_start = (start_frame / fps) + start_duration
            middle_end = (end_frame / fps) - end_duration
            middle_duration = middle_end - middle_start
            
            for i in range(middle_samples):
                timestamp = middle_start + (i * middle_duration / middle_samples)
                frame_idx = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append((pil_image, timestamp))
            
            # End samples
            end_start = (end_frame / fps) - end_duration
            for i in range(end_samples):
                timestamp = end_start + (i * end_duration / end_samples)
                frame_idx = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append((pil_image, timestamp))
        
        cap.release()
        return frames
    
    def read_text_from_frames(self, frame_data: List[tuple], custom_prompt: str = None) -> List[dict]:
        """
        Extract text from video frames using Qwen2.5-VL
        
        Args:
            frame_data: List of (PIL Image, timestamp) tuples
            custom_prompt: Custom prompt for text extraction
        
        Returns:
            List of dictionaries with text and timestamp info
        """
        if custom_prompt is None:
            prompt = "Read and transcribe all visible text in this image. Include any text from signs, captions, subtitles, or written content."
        else:
            prompt = custom_prompt
        
        results = []
        
        for i, (frame, timestamp) in enumerate(frame_data):
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            print(f"Processing frame {i+1}/{len(frame_data)} at {minutes:02d}:{seconds:02d}...")
            
            # Prepare messages for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": frame},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            inputs = inputs.to(self.device)
            
            # Generate response with fixed generation parameters
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    # Remove conflicting parameters
                )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
                results.append({
                    'text': output_text.strip(),
                    'timestamp': timestamp,
                    'time_formatted': f"{minutes:02d}:{seconds:02d}",
                    'frame_index': i
                })
        
        return results
    
    def process_video(self, video_path: str, max_frames: int = 20, 
                     extraction_method: str = "interval", interval_seconds: int = 10,
                     time_range: tuple = None, custom_prompt: str = None, 
                     batch_process: bool = False) -> dict:
        """
        Complete pipeline to extract text from video (optimized for long videos)
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to analyze
            extraction_method: How to extract frames ("uniform", "interval", "adaptive", "segments")
            interval_seconds: Seconds between frames for interval method (default: 10 seconds)
            time_range: (start_seconds, end_seconds) to process only part of video
            custom_prompt: Custom prompt for text extraction
            batch_process: Whether to process video in time segments
        
        Returns:
            Dictionary with frame texts, timestamps, and summary
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"Processing long video: {video_path}")
        
        if batch_process:
            return self._process_video_in_batches(video_path, max_frames, extraction_method, 
                                                interval_seconds, custom_prompt)
        
        # Extract frames with timestamps
        frame_data = self.extract_frames(video_path, max_frames, extraction_method, 
                                       interval_seconds, time_range)
        print(f"Extracted {len(frame_data)} frames")
        
        # Read text from frames
        frame_results = self.read_text_from_frames(frame_data, custom_prompt)
        
        # Process results
        all_texts = []
        unique_texts = set()
        timeline = []
        
        for result in frame_results[:]:
            text = result['text']
            if text and text not in unique_texts and len(text.strip()) > 0:
                unique_texts.add(text)
                timeline.append({
                    'time': result['time_formatted'],
                    'timestamp': result['timestamp'],
                    'text': text
                })
                all_texts.append(f"[{result['time_formatted']}] {text}")
        
        return {
            "total_frames_processed": len(frame_data),
            "extraction_method": extraction_method,
            "timeline": timeline,
            "unique_texts": list(unique_texts),
            "chronological_text": "\n".join(all_texts),
            "summary_stats": {
                "unique_text_segments": len(unique_texts),
                "processing_duration": f"{len(frame_data) * 2:.1f} seconds estimated",
                "video_coverage": f"{(len(frame_data) * interval_seconds) / 60:.1f} minutes sampled"
            }
        }
    
    def _process_video_in_batches(self, video_path: str, max_frames: int, 
                                method: str, interval_seconds: int, custom_prompt: str) -> dict:
        """
        Process very long videos in time-based batches to manage memory
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_seconds = total_frames / fps
        cap.release()
        
        # Process in 30-minute batches
        batch_duration = 30 * 60  # 30 minutes
        num_batches = int(np.ceil(duration_seconds / batch_duration))
        
        all_results = []
        
        print(f"Processing {duration_seconds/60:.1f} minute video in {num_batches} batches...")
        
        for batch_idx in range(num_batches):
            start_time = batch_idx * batch_duration
            end_time = min((batch_idx + 1) * batch_duration, duration_seconds)
            
            print(f"\nBatch {batch_idx + 1}/{num_batches}: {start_time/60:.1f}-{end_time/60:.1f} minutes")
            
            batch_results = self.process_video(
                video_path=video_path,
                max_frames=max_frames,
                extraction_method=method,
                interval_seconds=interval_seconds,
                time_range=(start_time, end_time),
                custom_prompt=custom_prompt,
                batch_process=False  # Prevent recursive batching
            )
            
            all_results.append(batch_results)
        
        # Combine all batch results
        combined_timeline = []
        all_unique_texts = set()
        
        for batch_result in all_results:
            combined_timeline.extend(batch_result['timeline'])
            all_unique_texts.update(batch_result['unique_texts'])
        
        # Sort timeline by timestamp
        combined_timeline.sort(key=lambda x: x['timestamp'])
        
        chronological_text = "\n".join([f"[{item['time']}] {item['text']}" 
                                       for item in combined_timeline])
        
        return {
            "total_frames_processed": sum(r["total_frames_processed"] for r in all_results),
            "extraction_method": f"{method} (batched)",
            "timeline": combined_timeline,
            "unique_texts": list(all_unique_texts),
            "chronological_text": chronological_text,
            "summary_stats": {
                "unique_text_segments": len(all_unique_texts),
                "total_batches": num_batches,
                "video_duration_minutes": duration_seconds / 60
            }
        }

# Example usage for frame every 10 seconds
def main():
    # Initialize the reader
    reader = QwenVideoTextReader()
    
    # For a 2h57 video, extract frame every 10 seconds
    video_path = "test.mp4"  # Replace with your video path
    
    try:
        print("=== EXTRACTING FRAME EVERY 10 SECONDS ===")
        # Calculate max_frames based on video duration and 10-second intervals
        # For 177.9 minutes (10674 seconds), we'd get ~1067 frames
        # You might want to limit this for processing time
        
        results = reader.process_video(
            video_path=video_path,
            max_frames=20,  # This will be limited by the actual video duration
            extraction_method="interval",
            interval_seconds=10,  # Frame every 10 seconds,
            custom_prompt="Extract all visible amazigh text including subtitles, titles, and any written content."
        )
        
        print(f"Processed {results['total_frames_processed']} frames")
        print(f"Found {results['summary_stats']['unique_text_segments']} unique text segments")
        print(f"Video coverage: {results['summary_stats']['video_coverage']}")
        
        print("\n=== FIRST 10 TIMELINE RESULTS ===")
        # Show first 10 chronological text entries with timestamps
        for item in results['timeline']:
            print(f"{item['time']}: {item['text'][:100]}...")
        
        # Save results to file
        with open("video_text_extraction_10sec.txt", "w", encoding='utf-8') as f:
            f.write("=== VIDEO TEXT EXTRACTION - FRAME EVERY 10 SECONDS ===\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Processing method: {results['extraction_method']}\n")
            f.write(f"Interval: 10 seconds\n")
            f.write(f"Total frames processed: {results['total_frames_processed']}\n")
            f.write(f"Total unique text segments: {len(results['unique_texts'])}\n\n")
            
            f.write("=== CHRONOLOGICAL TEXT ===\n")
            f.write(results['chronological_text'])
            
            f.write("\n\n=== UNIQUE TEXTS (DEDUPLICATED) ===\n")
            for i, text in enumerate(results['unique_texts'], 1):
                f.write(f"{i}. {text}\n")
        
        print("\nResults saved to 'video_text_extraction_10sec.txt'")
        
        # Optional: If you want to process in smaller batches to manage memory
        print("\n=== ALTERNATIVE: BATCH PROCESSING ===")
        # batch_results = reader.process_video(
        #     video_path=video_path,
        #     max_frames=50,  # 50 frames per 30-minute batch
        #     extraction_method="interval",
        #     interval_seconds=10,
        #     batch_process=True,
        #     custom_prompt="Extract all visible text including subtitles, titles, and any written content."
        # )
        
        # print(f"Batch processing completed: {batch_results['summary_stats']['unique_text_segments']} total text segments")
        
    except Exception as e:
        print(f"Error processing video: {e}")

# Utility function for extracting every 10 seconds with custom time range
def extract_10sec_intervals(video_path: str, start_minutes: int = 0, end_minutes: int = None, 
                           output_file: str = "text_10sec_intervals.txt"):
    """Extract text from video at 10-second intervals within specified time range"""
    reader = QwenVideoTextReader()
    
    # Convert minutes to seconds
    start_seconds = start_minutes * 60
    end_seconds = end_minutes * 60 if end_minutes else None
    
    # Calculate expected number of frames
    if end_seconds:
        duration = end_seconds - start_seconds
        expected_frames = duration // 10
        print(f"Processing {start_minutes}-{end_minutes} minutes, expecting ~{expected_frames} frames")
    
    results = reader.process_video(
        video_path=video_path,
        max_frames=2000,  # High limit, will be constrained by actual video length
        extraction_method="interval",
        interval_seconds=10,
        time_range=(start_seconds, end_seconds) if end_seconds else None,
        custom_prompt="Read all text visible in this frame including subtitles, captions, and any written content."
    )
    
    # Save results
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(f"=== TEXT EXTRACTION EVERY 10 SECONDS ===\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Time range: {start_minutes} - {end_minutes or 'end'} minutes\n")
        f.write(f"Total frames: {results['total_frames_processed']}\n")
        f.write(f"Unique text segments: {len(results['unique_texts'])}\n\n")
        
        f.write("=== TIMELINE ===\n")
        for item in results['timeline']:
            f.write(f"[{item['time']}] {item['text']}\n")
    
    print(f"Results saved to {output_file}")
    return results

if __name__ == "__main__":
    main()