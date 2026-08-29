/**
 * Audio Extractor Module (Browser Web Audio API)
 * Extracts audio tracks locally from video files and encodes them to 16kHz mono WAV,
 * reducing multi-gigabyte video uploads to small megabyte payloads for AI processing.
 */

export class ClientAudioExtractor {
  /**
   * Extracts or prepares audio from a user-selected File.
   * @param {File} file - Original video or audio file.
   * @param {(status: string, progress: number) => void} onProgress - Progress callback.
   * @returns {Promise<Blob>} - 16kHz mono WAV Blob (or original audio file).
   */
  static async extractAudioBlob(file, onProgress = () => {}) {
    // If user already supplied an audio file, send it directly
    if (file.type.startsWith("audio/")) {
      onProgress("Arquivo de áudio detectado, pronto para envio...", 100);
      return file;
    }

    onProgress("Lendo arquivo de vídeo local na memória...", 15);
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000,
    });

    try {
      onProgress("Carregando buffer de dados...", 30);
      const arrayBuffer = await file.arrayBuffer();

      onProgress("Decodificando faixa de áudio com Web Audio API...", 55);
      const decodedBuffer = await audioContext.decodeAudioData(arrayBuffer);

      onProgress("Convertendo áudio para 16kHz mono (otimizado para IA)...", 80);
      const wavBlob = this.encodeWav(decodedBuffer, 16000);

      onProgress("Faixa de áudio extraída com sucesso!", 100);
      return wavBlob;
    } catch (err) {
      console.warn("Extração Web Audio API falhou ou formato não decodificável:", err);
      onProgress("Usando upload direto do arquivo como fallback...", 100);
      // Return the original file if local extraction fails
      return file;
    } finally {
      if (audioContext && audioContext.state !== "closed") {
        audioContext.close().catch(() => {});
      }
    }
  }

  /**
   * Encodes an AudioBuffer into a standard 16-bit PCM 16kHz mono WAV Blob.
   * @param {AudioBuffer} buffer
   * @param {number} targetSampleRate
   * @returns {Blob}
   */
  static encodeWav(buffer, targetSampleRate = 16000) {
    const numChannels = buffer.numberOfChannels;
    const sourceSampleRate = buffer.sampleRate;

    // Mix down all channels to mono
    const length = Math.round((buffer.length * targetSampleRate) / sourceSampleRate);
    const monoSamples = new Float32Array(length);

    const sourceData = [];
    for (let c = 0; c < numChannels; c++) {
      sourceData.push(buffer.getChannelData(c));
    }

    // Linear interpolation resampling & mixing
    const ratio = sourceSampleRate / targetSampleRate;
    for (let i = 0; i < length; i++) {
      const sourceIdx = i * ratio;
      const idxFloor = Math.floor(sourceIdx);
      const idxCeil = Math.min(buffer.length - 1, idxFloor + 1);
      const weight = sourceIdx - idxFloor;

      let mixed = 0;
      for (let c = 0; c < numChannels; c++) {
        const val = sourceData[c][idxFloor] * (1 - weight) + sourceData[c][idxCeil] * weight;
        mixed += val;
      }
      monoSamples[i] = mixed / numChannels;
    }

    // Create 16-bit PCM WAV ArrayBuffer (Header 44 bytes + PCM data)
    const pcmByteLength = monoSamples.length * 2;
    const wavBuffer = new ArrayBuffer(44 + pcmByteLength);
    const view = new DataView(wavBuffer);

    // RIFF identifier
    this.writeString(view, 0, "RIFF");
    // file length minus RIFF identifier and length = 36 + pcmByteLength
    view.setUint32(4, 36 + pcmByteLength, true);
    // RIFF type
    this.writeString(view, 8, "WAVE");
    // format chunk identifier
    this.writeString(view, 12, "fmt ");
    // format chunk length
    view.setUint32(16, 16, true);
    // sample format (raw PCM)
    view.setUint16(20, 1, true);
    // channel count (1 = mono)
    view.setUint16(22, 1, true);
    // sample rate
    view.setUint32(24, targetSampleRate, true);
    // byte rate (sample rate * block align)
    view.setUint32(28, targetSampleRate * 2, true);
    // block align (channel count * bytes per sample)
    view.setUint16(32, 2, true);
    // bits per sample
    view.setUint16(34, 16, true);
    // data chunk identifier
    this.writeString(view, 36, "data");
    // data chunk length
    view.setUint32(40, pcmByteLength, true);

    // Write PCM samples (clamped to [-1, 1])
    let offset = 44;
    for (let i = 0; i < monoSamples.length; i++) {
      let s = Math.max(-1, Math.min(1, monoSamples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }

    return new Blob([wavBuffer], { type: "audio/wav" });
  }

  static writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }
}
