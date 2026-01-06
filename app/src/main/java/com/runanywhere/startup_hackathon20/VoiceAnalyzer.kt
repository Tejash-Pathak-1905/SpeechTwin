package com.runanywhere.startup_hackathon20

import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

class VoiceAnalyzer {
    companion object {
        private const val TAG = "VoiceAnalyzer"
        private const val SAMPLE_RATE = 44100
        private const val FRAME_SIZE_MS = 50
        private const val FRAME_SIZE = (SAMPLE_RATE * FRAME_SIZE_MS) / 1000
        private const val HOP_SIZE = FRAME_SIZE / 2
    }

    // Keep old analyze() signature for backward compatibility
    data class AnalysisResult(
        val pitch: Float,
        val loudness: Float,
        val jitter: Float,
        val shimmer: Float,
        val healthScore: Int,
        val duration: Int = 10
    )

    fun analyze(filePath: String): AnalysisResult {
        val features = analyzeComplete(filePath)
        return AnalysisResult(
            pitch = features.pitch,
            loudness = features.loudness,
            jitter = features.jitter,
            shimmer = features.shimmer,
            healthScore = features.healthScore,
            duration = features.duration.toInt()
        )
    }

    // NEW: Complete analysis with all 22 features
    fun analyzeComplete(filePath: String): VoiceFeatures {
        Log.d(TAG, "=== Starting Complete Voice Analysis ===")
        Log.d(TAG, "File: $filePath")

        val file = File(filePath)
        if (!file.exists() || !file.canRead()) {
            Log.e(TAG, "File error")
            return createDefaultFeatures(filePath)
        }

        try {
            val audioData = loadWavFile(file)
            if (audioData.isEmpty()) return createDefaultFeatures(filePath)

            val duration = audioData.size.toFloat() / SAMPLE_RATE
            Log.d(TAG, "Loaded ${audioData.size} samples, ${duration}s")

            val normalized = normalizeAudio(audioData)
            return extractAllFeatures(normalized, filePath, duration)

        } catch (e: Exception) {
            Log.e(TAG, "Analysis error: ${e.message}", e)
            return createDefaultFeatures(filePath)
        }
    }

    private fun extractAllFeatures(audio: FloatArray, filePath: String, duration: Float): VoiceFeatures {
        val frames = createFrames(audio)

        val pitchValues = mutableListOf<Float>()
        val amplitudeValues = mutableListOf<Float>()
        val energyValues = mutableListOf<Float>()
        val zcrValues = mutableListOf<Float>()
        val hnrValues = mutableListOf<Float>()
        val spectralCentroids = mutableListOf<Float>()
        val spectralRolloffs = mutableListOf<Float>()
        val spectralFluxes = mutableListOf<Float>()
        var previousSpectrum: FloatArray? = null
        var voicedFrames = 0

        // Analyze each frame
        for (frame in frames) {
            val windowed = applyHammingWindow(frame)

            val pitch = calculatePitchForFrame(windowed)
            pitchValues.add(pitch)
            if (pitch > 0) voicedFrames++

            val amp = calculateRMS(windowed)
            amplitudeValues.add(amp)

            energyValues.add(calculateEnergy(windowed))
            zcrValues.add(calculateZCR(windowed))

            if (pitch > 0) {
                hnrValues.add(calculateHNR(windowed, pitch))
            }

            val fftResult = fft(windowed)
            val magSpec = getMagnitudeSpectrum(fftResult)

            spectralCentroids.add(calculateSpectralCentroid(magSpec))
            spectralRolloffs.add(calculateSpectralRolloff(magSpec))

            previousSpectrum?.let {
                spectralFluxes.add(calculateSpectralFlux(it, magSpec))
            }
            previousSpectrum = magSpec
        }

        // Calculate aggregate features
        val validPitch = pitchValues.filter { it > 0 }
        val pitch = if (validPitch.isNotEmpty()) validPitch.sorted()[validPitch.size / 2] else 180f
        val pitchMin = validPitch.minOrNull() ?: 0f
        val pitchMax = validPitch.maxOrNull() ?: 0f

        val loudness = calculateLoudness(audio)
        val loudnessMin = amplitudeValues.minOrNull()?.let { 20 * log10(it + 1e-10f) } ?: -60f
        val loudnessMax = amplitudeValues.maxOrNull()?.let { 20 * log10(it + 1e-10f) } ?: -60f

        val jitter = calculateJitter(validPitch)
        val shimmer = calculateShimmer(amplitudeValues.filter { it > 0.001f })
        val hnr = if (hnrValues.isNotEmpty()) hnrValues.average().toFloat() else 20f
        val zcr = zcrValues.average().toFloat()
        val energy = energyValues.average().toFloat()
        val entropy = calculateSpectralEntropy(audio)

        val formants = extractFormants(audio)
        val formantF1 = formants.getOrNull(0) ?: 700f
        val formantF2 = formants.getOrNull(1) ?: 1220f
        val formantF3 = formants.getOrNull(2) ?: 2600f

        val spectralCentroid = spectralCentroids.average().toFloat()
        val spectralRolloff = spectralRolloffs.average().toFloat()
        val spectralFlux = if (spectralFluxes.isNotEmpty()) spectralFluxes.average().toFloat() else 0f

        val breathiness = calculateBreathiness(hnr, spectralCentroid)
        val roughness = calculateRoughness(jitter, shimmer, hnr)
        val tremor = calculateTremor(validPitch, amplitudeValues)

        val pauses = detectPauses(energyValues)
        val pauseCount = pauses.size
        val pauseDuration = pauses.sumOf { it.second - it.first }.toFloat() * FRAME_SIZE_MS / 1000f

        val voicedRatio = voicedFrames.toFloat() / frames.size
        val articulationRate = voicedRatio
        val tempo = calculateTempo(pauses, duration, voicedRatio)

        val mfcc = calculateAverageMFCC(frames)
        val chromagram = calculateAverageChromagram(frames)

        val healthScore = calculateHealthScore(pitch, loudness, jitter, shimmer, hnr, breathiness, roughness, tremor)

        val recordingId = "REC_${System.currentTimeMillis()}"

        return VoiceFeatures(
            recordingId = recordingId,
            timestamp = System.currentTimeMillis(),
            filePath = filePath,
            duration = duration,
            pitch = pitch,
            loudness = loudness,
            jitter = jitter,
            shimmer = shimmer,
            hnr = hnr,
            zcr = zcr,
            energy = energy,
            entropy = entropy,
            tempo = tempo,
            formantF1 = formantF1,
            formantF2 = formantF2,
            formantF3 = formantF3,
            spectralCentroid = spectralCentroid,
            spectralRolloff = spectralRolloff,
            spectralFlux = spectralFlux,
            breathiness = breathiness,
            roughness = roughness,
            tremor = tremor,
            pauseCount = pauseCount,
            pauseDuration = pauseDuration,
            articulationRate = articulationRate,
            mfcc = mfcc,
            chromagram = chromagram,
            healthScore = healthScore,
            pitchRange = Pair(pitchMin, pitchMax),
            loudnessRange = Pair(loudnessMin, loudnessMax),
            voicedFramesRatio = voicedRatio
        )
    }

    // ===== DSP Helper Functions =====

    private fun createFrames(audio: FloatArray): List<FloatArray> {
        val frames = mutableListOf<FloatArray>()
        var start = 0
        while (start + FRAME_SIZE <= audio.size) {
            frames.add(audio.copyOfRange(start, start + FRAME_SIZE))
            start += HOP_SIZE
        }
        return frames
    }

    private fun applyHammingWindow(signal: FloatArray): FloatArray {
        val n = signal.size
        return FloatArray(n) { i ->
            val window = 0.54f - 0.46f * cos(2f * PI.toFloat() * i / (n - 1))
            signal[i] * window
        }
    }

    private fun calculatePitchForFrame(frame: FloatArray): Float {
        val minPeriod = SAMPLE_RATE / 400
        val maxPeriod = SAMPLE_RATE / 80
        val energy = frame.map { it * it }.average()
        if (energy < 0.001) return 0f

        var maxCorr = 0f
        var bestPeriod = 0
        for (period in minPeriod..minOf(maxPeriod, frame.size / 2)) {
            var corr = 0f
            var count = 0
            for (i in 0 until frame.size - period) {
                corr += frame[i] * frame[i + period]
                count++
            }
            if (count > 0) {
                corr /= count
                if (corr > maxCorr) {
                    maxCorr = corr
                    bestPeriod = period
                }
            }
        }
        return if (bestPeriod > 0 && maxCorr > 0.3f) SAMPLE_RATE.toFloat() / bestPeriod else 0f
    }

    private fun calculateRMS(frame: FloatArray): Float {
        return sqrt(frame.map { it * it.toDouble() }.average()).toFloat()
    }

    private fun calculateEnergy(frame: FloatArray): Float {
        return frame.map { it * it }.sum() / frame.size
    }

    private fun calculateZCR(frame: FloatArray): Float {
        var crossings = 0
        for (i in 1 until frame.size) {
            if ((frame[i] >= 0 && frame[i - 1] < 0) || (frame[i] < 0 && frame[i - 1] >= 0)) {
                crossings++
            }
        }
        return crossings.toFloat()
    }

    private fun calculateHNR(frame: FloatArray, pitch: Float): Float {
        val period = (SAMPLE_RATE / pitch).toInt()
        if (period <= 0 || period >= frame.size / 2) return 0f

        var harmonicEnergy = 0f
        var noiseEnergy = 0f
        var count = 0

        for (i in 0 until frame.size - period) {
            val harmonic = (frame[i] + frame[i + period]) / 2f
            val noise = frame[i] - harmonic
            harmonicEnergy += harmonic * harmonic
            noiseEnergy += noise * noise
            count++
        }

        return if (noiseEnergy > 1e-10f && count > 0) {
            10 * log10((harmonicEnergy / noiseEnergy).toDouble()).toFloat()
        } else 30f
    }

    private fun fft(signal: FloatArray): Array<Pair<Float, Float>> {
        val n = 2.0.pow(ceil(log2(signal.size.toDouble()))).toInt()
        val padded = signal + FloatArray(n - signal.size) { 0f }
        return fftRecursive(padded.map { Pair(it, 0f) }.toTypedArray())
    }

    private fun fftRecursive(x: Array<Pair<Float, Float>>): Array<Pair<Float, Float>> {
        val n = x.size
        if (n <= 1) return x

        val even = x.filterIndexed { i, _ -> i % 2 == 0 }.toTypedArray()
        val odd = x.filterIndexed { i, _ -> i % 2 == 1 }.toTypedArray()
        val evenFFT = fftRecursive(even)
        val oddFFT = fftRecursive(odd)

        val result = Array(n) { Pair(0f, 0f) }
        for (k in 0 until n / 2) {
            val angle = -2.0 * PI * k / n
            val wReal = cos(angle).toFloat()
            val wImag = sin(angle).toFloat()
            val tReal = wReal * oddFFT[k].first - wImag * oddFFT[k].second
            val tImag = wReal * oddFFT[k].second + wImag * oddFFT[k].first
            result[k] = Pair(evenFFT[k].first + tReal, evenFFT[k].second + tImag)
            result[k + n / 2] = Pair(evenFFT[k].first - tReal, evenFFT[k].second - tImag)
        }
        return result
    }

    private fun getMagnitudeSpectrum(fft: Array<Pair<Float, Float>>): FloatArray {
        return fft.map { (r, i) -> sqrt(r * r + i * i) }.toFloatArray()
    }

    private fun calculateSpectralCentroid(spectrum: FloatArray): Float {
        var weightedSum = 0f
        var sum = 0f
        for (i in spectrum.indices) {
            val freq = i * SAMPLE_RATE.toFloat() / spectrum.size
            weightedSum += freq * spectrum[i]
            sum += spectrum[i]
        }
        return if (sum > 0) weightedSum / sum else 0f
    }

    private fun calculateSpectralRolloff(spectrum: FloatArray): Float {
        val totalEnergy = spectrum.sum()
        val threshold = 0.85f * totalEnergy
        var cumEnergy = 0f
        for (i in spectrum.indices) {
            cumEnergy += spectrum[i]
            if (cumEnergy >= threshold) {
                return i * SAMPLE_RATE.toFloat() / spectrum.size
            }
        }
        return 0f
    }

    private fun calculateSpectralFlux(prev: FloatArray, curr: FloatArray): Float {
        val minSize = minOf(prev.size, curr.size)
        var flux = 0f
        for (i in 0 until minSize) {
            val diff = curr[i] - prev[i]
            flux += diff * diff
        }
        return sqrt(flux / minSize)
    }

    private fun calculateSpectralEntropy(audio: FloatArray): Float {
        val frameSize = 2048
        if (audio.size < frameSize) return 0f

        val frame = audio.copyOfRange(0, frameSize)
        val windowed = applyHammingWindow(frame)
        val fftResult = fft(windowed)
        val powerSpectrum = fftResult.map { (r, i) -> r * r + i * i }

        val totalPower = powerSpectrum.sum()
        if (totalPower < 1e-10f) return 0f

        var entropy = 0f
        for (p in powerSpectrum) {
            val prob = p / totalPower
            if (prob > 1e-10f) {
                entropy -= prob * ln(prob)
            }
        }
        return (entropy / ln(powerSpectrum.size.toFloat())).coerceIn(0f, 1f)
    }

    private fun extractFormants(audio: FloatArray): List<Float> {
        val frameSize = 2048
        if (audio.size < frameSize) return listOf(700f, 1220f, 2600f)

        val frame = audio.copyOfRange(0, frameSize)
        val lpcOrder = 12
        val lpcCoeffs = calculateLPC(frame, lpcOrder)

        // Find resonance peaks (formants)
        val formants = mutableListOf<Float>()
        val fftSize = 512
        for (k in 1 until fftSize / 2) {
            val freq = k * SAMPLE_RATE.toFloat() / fftSize
            if (freq < 200 || freq > 4000) continue

            val omega = 2 * PI * freq / SAMPLE_RATE
            var real = 1.0
            var imag = 0.0
            for (i in 1 until lpcCoeffs.size) {
                val angle = omega * i
                real += lpcCoeffs[i] * cos(angle)
                imag += lpcCoeffs[i] * sin(angle)
            }
            val magnitude = 1.0 / sqrt(real * real + imag * imag)

            if (magnitude > 0.5 && formants.size < 3) {
                if (formants.isEmpty() || abs(formants.last() - freq) > 400) {
                    formants.add(freq)
                }
            }
        }

        while (formants.size < 3) {
            formants.add(listOf(700f, 1220f, 2600f)[formants.size])
        }

        return formants.take(3)
    }

    private fun calculateLPC(signal: FloatArray, order: Int): FloatArray {
        val n = signal.size
        val r = FloatArray(order + 1) { 0f }

        for (i in 0..order) {
            for (j in 0 until n - i) {
                r[i] += signal[j] * signal[j + i]
            }
        }

        val a = FloatArray(order + 1) { 0f }
        val e = FloatArray(order + 1) { 0f }
        a[0] = 1f
        e[0] = r[0]

        for (i in 1..order) {
            var lambda = 0f
            for (j in 0 until i) {
                lambda += a[j] * r[i - j]
            }
            lambda /= -e[i - 1]

            val aPrev = a.copyOf()
            for (j in 0 until i) {
                a[j] = aPrev[j] + lambda * aPrev[i - 1 - j]
            }
            a[i] = lambda
            e[i] = (1 - lambda * lambda) * e[i - 1]
        }

        return a
    }

    private fun calculateBreathiness(hnr: Float, spectralCentroid: Float): Float {
        val hnrFactor = (30f - hnr.coerceIn(0f, 30f)) / 30f
        val centroidFactor = (spectralCentroid - 1000f).coerceIn(0f, 2000f) / 2000f
        return ((hnrFactor + centroidFactor) / 2f).coerceIn(0f, 1f)
    }

    private fun calculateRoughness(jitter: Float, shimmer: Float, hnr: Float): Float {
        val jitterFactor = (jitter / 5f).coerceIn(0f, 1f)
        val shimmerFactor = (shimmer / 10f).coerceIn(0f, 1f)
        val hnrFactor = (25f - hnr.coerceIn(0f, 25f)) / 25f
        return ((jitterFactor + shimmerFactor + hnrFactor) / 3f).coerceIn(0f, 1f)
    }

    private fun calculateTremor(pitchValues: List<Float>, amplitudeValues: List<Float>): Float {
        if (pitchValues.size < 10) return 0f

        val pitchVariations = mutableListOf<Float>()
        for (i in 1 until pitchValues.size) {
            pitchVariations.add(abs(pitchValues[i] - pitchValues[i - 1]))
        }

        val ampVariations = mutableListOf<Float>()
        for (i in 1 until amplitudeValues.size) {
            ampVariations.add(abs(amplitudeValues[i] - amplitudeValues[i - 1]))
        }

        val pitchTremor = if (pitchVariations.isNotEmpty()) pitchVariations.average().toFloat() / 50f else 0f
        val ampTremor = if (ampVariations.isNotEmpty()) ampVariations.average().toFloat() * 10f else 0f

        return ((pitchTremor + ampTremor) / 2f).coerceIn(0f, 1f)
    }

    private fun detectPauses(energyValues: List<Float>): List<Pair<Int, Int>> {
        val threshold = energyValues.average() * 0.1f
        val pauses = mutableListOf<Pair<Int, Int>>()
        var pauseStart = -1

        for (i in energyValues.indices) {
            if (energyValues[i] < threshold) {
                if (pauseStart == -1) pauseStart = i
            } else {
                if (pauseStart != -1 && i - pauseStart >= 3) {
                    pauses.add(Pair(pauseStart, i))
                }
                pauseStart = -1
            }
        }

        return pauses
    }

    private fun calculateTempo(pauses: List<Pair<Int, Int>>, duration: Float, voicedRatio: Float): Float {
        val syllablesPerSecond = (voicedRatio * 4f).coerceIn(1f, 8f)
        return syllablesPerSecond
    }

    private fun calculateAverageMFCC(frames: List<FloatArray>): List<Float> {
        val numCoeffs = 13
        val mfccList = mutableListOf<FloatArray>()

        for (i in 0 until minOf(frames.size, 20)) {
            val mfcc = calculateMFCC(frames[i], numCoeffs)
            mfccList.add(mfcc)
        }

        val avgMFCC = FloatArray(numCoeffs) { 0f }
        for (mfcc in mfccList) {
            for (i in 0 until numCoeffs) {
                avgMFCC[i] += mfcc[i]
            }
        }
        for (i in 0 until numCoeffs) {
            avgMFCC[i] /= mfccList.size
        }

        return avgMFCC.toList()
    }

    private fun calculateMFCC(frame: FloatArray, numCoeffs: Int): FloatArray {
        val preEmphasis = 0.97f
        val emphasized = FloatArray(frame.size)
        emphasized[0] = frame[0]
        for (i in 1 until frame.size) {
            emphasized[i] = frame[i] - preEmphasis * frame[i - 1]
        }

        val windowed = applyHammingWindow(emphasized)
        val fftResult = fft(windowed)
        val powerSpectrum = fftResult.map { (r, i) -> r * r + i * i }.toFloatArray()

        val numFilters = 26
        val filterBank = createMelFilterBank(numFilters, windowed.size)
        val melEnergies = FloatArray(numFilters) { 0f }

        for (i in 0 until numFilters) {
            for (k in 0 until minOf(filterBank[i].size, powerSpectrum.size)) {
                melEnergies[i] += filterBank[i][k] * powerSpectrum[k]
            }
            melEnergies[i] = ln(melEnergies[i] + 1e-10f)
        }

        val mfcc = FloatArray(numCoeffs) { 0f }
        for (i in 0 until numCoeffs) {
            for (j in 0 until numFilters) {
                mfcc[i] += melEnergies[j] * cos(PI * i * (j + 0.5) / numFilters).toFloat()
            }
        }

        return mfcc
    }

    private fun createMelFilterBank(numFilters: Int, fftSize: Int): Array<FloatArray> {
        val lowFreqMel = hzToMel(0f)
        val highFreqMel = hzToMel(SAMPLE_RATE / 2f)
        val melPoints = FloatArray(numFilters + 2) { i ->
            lowFreqMel + (highFreqMel - lowFreqMel) * i / (numFilters + 1)
        }
        val hzPoints = melPoints.map { melToHz(it) }
        val binPoints = hzPoints.map { (it / SAMPLE_RATE * fftSize).toInt() }

        val filterBank = Array(numFilters) { FloatArray(fftSize / 2 + 1) { 0f } }
        for (i in 0 until numFilters) {
            val leftBin = binPoints[i]
            val centerBin = binPoints[i + 1]
            val rightBin = binPoints[i + 2]

            for (k in leftBin until centerBin) {
                if (k < filterBank[i].size) {
                    filterBank[i][k] = (k - leftBin).toFloat() / (centerBin - leftBin)
                }
            }
            for (k in centerBin until rightBin) {
                if (k < filterBank[i].size) {
                    filterBank[i][k] = (rightBin - k).toFloat() / (rightBin - centerBin)
                }
            }
        }
        return filterBank
    }

    private fun hzToMel(hz: Float): Float = 2595f * log10(1f + hz / 700f)
    private fun melToHz(mel: Float): Float = 700f * (10f.pow(mel / 2595f) - 1f)

    private fun calculateAverageChromagram(frames: List<FloatArray>): List<Float> {
        val chromaList = mutableListOf<FloatArray>()
        for (i in 0 until minOf(frames.size, 20)) {
            chromaList.add(calculateChromagram(frames[i]))
        }

        val avgChroma = FloatArray(12) { 0f }
        for (chroma in chromaList) {
            for (i in 0 until 12) {
                avgChroma[i] += chroma[i]
            }
        }
        for (i in 0 until 12) {
            avgChroma[i] /= chromaList.size
        }
        return avgChroma.toList()
    }

    private fun calculateChromagram(frame: FloatArray): FloatArray {
        val fftResult = fft(frame)
        val magSpec = getMagnitudeSpectrum(fftResult)
        val chroma = FloatArray(12) { 0f }

        for (k in 1 until magSpec.size / 2) {
            val freq = k * SAMPLE_RATE.toFloat() / frame.size
            if (freq < 20 || freq > 5000) continue
            val midiNote = 12 * log2(freq / 440.0) + 69
            if (midiNote < 0) continue
            val pitchClass = (midiNote.toInt() % 12)
            chroma[pitchClass] += magSpec[k]
        }

        val maxVal = chroma.maxOrNull() ?: 1f
        if (maxVal > 0) {
            for (i in chroma.indices) chroma[i] /= maxVal
        }
        return chroma
    }

    private fun calculateLoudness(audio: FloatArray): Float {
        val rms = sqrt(audio.map { it * it.toDouble() }.average())
        return if (rms > 0.0) 20 * log10(rms).toFloat() else -60f
    }

    private fun calculateJitter(pitchValues: List<Float>): Float {
        if (pitchValues.size < 3) return 0f
        val mean = pitchValues.average().toFloat()
        val variance = pitchValues.map { (it - mean) * (it - mean) }.average()
        val stddev = sqrt(variance).toFloat()
        return (stddev / mean * 100f).coerceIn(0f, 100f)
    }

    private fun calculateShimmer(ampValues: List<Float>): Float {
        if (ampValues.size < 3) return 0f
        val mean = ampValues.average().toFloat()
        val variance = ampValues.map { (it - mean) * (it - mean) }.average()
        val stddev = sqrt(variance).toFloat()
        return (stddev / mean * 100f).coerceIn(0f, 100f)
    }

    private fun calculateHealthScore(
        pitch: Float, loudness: Float, jitter: Float, shimmer: Float,
        hnr: Float, breathiness: Float, roughness: Float, tremor: Float
    ): Int {
        var score = 100

        if (pitch < 80 || pitch > 400) score -= 20
        else if (pitch < 100 || pitch > 300) score -= 15
        else if (pitch < 120 || pitch > 250) score -= 5

        if (loudness < -50) score -= 20
        else if (loudness < -40) score -= 15
        else if (loudness < -30) score -= 5
        else if (loudness > -5) score -= 10

        if (jitter > 5.0f) score -= ((jitter - 2f) * 5).toInt().coerceAtMost(30)
        else if (jitter > 2.0f) score -= ((jitter - 2f) * 10).toInt()

        if (shimmer > 10.0f) score -= ((shimmer - 5f) * 3).toInt().coerceAtMost(25)
        else if (shimmer > 5.0f) score -= ((shimmer - 5f) * 10).toInt()

        if (hnr < 10f) score -= 20
        else if (hnr < 15f) score -= 10

        if (breathiness > 0.7f) score -= 15
        else if (breathiness > 0.5f) score -= 10

        if (roughness > 0.7f) score -= 15
        else if (roughness > 0.5f) score -= 10

        if (tremor > 0.3f) score -= 15
        else if (tremor > 0.2f) score -= 10

        return score.coerceIn(0, 100)
    }

    private fun loadWavFile(file: File): ShortArray {
        try {
            val inputStream = FileInputStream(file)
            val allBytes = inputStream.readBytes()
            inputStream.close()

            var dataOffset = 44
            for (i in 0 until minOf(200, allBytes.size - 4)) {
                if (allBytes[i] == 'd'.code.toByte() &&
                    allBytes[i + 1] == 'a'.code.toByte() &&
                    allBytes[i + 2] == 't'.code.toByte() &&
                    allBytes[i + 3] == 'a'.code.toByte()) {
                    dataOffset = i + 8
                    break
                }
            }

            val audioBytes = allBytes.copyOfRange(dataOffset, allBytes.size)
            val buffer = ByteBuffer.wrap(audioBytes).order(ByteOrder.LITTLE_ENDIAN)
            val samples = ShortArray(audioBytes.size / 2)
            for (i in samples.indices) {
                if (i * 2 + 1 < audioBytes.size) {
                    samples[i] = buffer.getShort(i * 2)
                }
            }
            return samples
        } catch (e: Exception) {
            Log.e(TAG, "Error loading WAV: ${e.message}", e)
            return shortArrayOf()
        }
    }

    private fun normalizeAudio(audio: ShortArray): FloatArray {
        val maxAmplitude = audio.maxOfOrNull { abs(it.toFloat()) } ?: 0f
        if (maxAmplitude < 100f) {
            return FloatArray(audio.size) { audio[it].toFloat() / 32768f }
        }
        val scaleFactor = 32000f / maxAmplitude
        return FloatArray(audio.size) { (audio[it].toFloat() * scaleFactor) / 32768f }
    }

    private fun createDefaultFeatures(filePath: String): VoiceFeatures {
        return VoiceFeatures(
            recordingId = "REC_${System.currentTimeMillis()}",
            timestamp = System.currentTimeMillis(),
            filePath = filePath,
            duration = 0f,
            pitch = 0f,
            loudness = -60f,
            jitter = 0f,
            shimmer = 0f,
            hnr = 0f,
            zcr = 0f,
            energy = 0f,
            entropy = 0f,
            tempo = 0f,
            formantF1 = 700f,
            formantF2 = 1220f,
            formantF3 = 2600f,
            spectralCentroid = 0f,
            spectralRolloff = 0f,
            spectralFlux = 0f,
            breathiness = 0f,
            roughness = 0f,
            tremor = 0f,
            pauseCount = 0,
            pauseDuration = 0f,
            articulationRate = 0f,
            mfcc = List(13) { 0f },
            chromagram = List(12) { 0f },
            healthScore = 0,
            pitchRange = Pair(0f, 0f),
            loudnessRange = Pair(-60f, -60f),
            voicedFramesRatio = 0f
        )
    }
}
