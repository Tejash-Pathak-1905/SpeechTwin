package com.runanywhere.startup_hackathon20

import java.io.Serializable

/**
 * Complete Voice Features Data Model - All 22+ features
 */
data class VoiceFeatures(
    val recordingId: String,
    val timestamp: Long,
    val filePath: String,
    val duration: Float,

    // Basic Features (4)
    val pitch: Float,
    val loudness: Float,
    val jitter: Float,
    val shimmer: Float,

    // Advanced Features (5)
    val hnr: Float,
    val zcr: Float,
    val energy: Float,
    val entropy: Float,
    val tempo: Float,

    // Formants (3)
    val formantF1: Float,
    val formantF2: Float,
    val formantF3: Float,

    // Spectral Features (3)
    val spectralCentroid: Float,
    val spectralRolloff: Float,
    val spectralFlux: Float,

    // Voice Quality (3)
    val breathiness: Float,
    val roughness: Float,
    val tremor: Float,

    // Temporal Features (3)
    val pauseCount: Int,
    val pauseDuration: Float,
    val articulationRate: Float,

    // MFCC & Chromagram
    val mfcc: List<Float>,
    val chromagram: List<Float>,

    // Health Score
    val healthScore: Int,

    // Statistics
    val pitchRange: Pair<Float, Float>,
    val loudnessRange: Pair<Float, Float>,
    val voicedFramesRatio: Float
) : Serializable {

    fun toMap(): Map<String, Any> = mapOf(
        "recordingId" to recordingId,
        "timestamp" to timestamp,
        "filePath" to filePath,
        "duration" to duration,
        "pitch" to pitch,
        "loudness" to loudness,
        "jitter" to jitter,
        "shimmer" to shimmer,
        "hnr" to hnr,
        "zcr" to zcr,
        "energy" to energy,
        "entropy" to entropy,
        "tempo" to tempo,
        "formantF1" to formantF1,
        "formantF2" to formantF2,
        "formantF3" to formantF3,
        "spectralCentroid" to spectralCentroid,
        "spectralRolloff" to spectralRolloff,
        "spectralFlux" to spectralFlux,
        "breathiness" to breathiness,
        "roughness" to roughness,
        "tremor" to tremor,
        "pauseCount" to pauseCount,
        "pauseDuration" to pauseDuration,
        "articulationRate" to articulationRate,
        "mfcc" to mfcc,
        "chromagram" to chromagram,
        "healthScore" to healthScore,
        "pitchRangeMin" to pitchRange.first,
        "pitchRangeMax" to pitchRange.second,
        "loudnessRangeMin" to loudnessRange.first,
        "loudnessRangeMax" to loudnessRange.second,
        "voicedFramesRatio" to voicedFramesRatio
    )

    companion object {
        fun fromMap(map: Map<String, Any>): VoiceFeatures {
            return VoiceFeatures(
                recordingId = map["recordingId"] as String,
                timestamp = (map["timestamp"] as Number).toLong(),
                filePath = map["filePath"] as String,
                duration = (map["duration"] as Number).toFloat(),
                pitch = (map["pitch"] as Number).toFloat(),
                loudness = (map["loudness"] as Number).toFloat(),
                jitter = (map["jitter"] as Number).toFloat(),
                shimmer = (map["shimmer"] as Number).toFloat(),
                hnr = (map["hnr"] as Number).toFloat(),
                zcr = (map["zcr"] as Number).toFloat(),
                energy = (map["energy"] as Number).toFloat(),
                entropy = (map["entropy"] as Number).toFloat(),
                tempo = (map["tempo"] as Number).toFloat(),
                formantF1 = (map["formantF1"] as Number).toFloat(),
                formantF2 = (map["formantF2"] as Number).toFloat(),
                formantF3 = (map["formantF3"] as Number).toFloat(),
                spectralCentroid = (map["spectralCentroid"] as Number).toFloat(),
                spectralRolloff = (map["spectralRolloff"] as Number).toFloat(),
                spectralFlux = (map["spectralFlux"] as Number).toFloat(),
                breathiness = (map["breathiness"] as Number).toFloat(),
                roughness = (map["roughness"] as Number).toFloat(),
                tremor = (map["tremor"] as Number).toFloat(),
                pauseCount = (map["pauseCount"] as Number).toInt(),
                pauseDuration = (map["pauseDuration"] as Number).toFloat(),
                articulationRate = (map["articulationRate"] as Number).toFloat(),
                mfcc = (map["mfcc"] as List<*>).map { (it as Number).toFloat() },
                chromagram = (map["chromagram"] as List<*>).map { (it as Number).toFloat() },
                healthScore = (map["healthScore"] as Number).toInt(),
                pitchRange = Pair(
                    (map["pitchRangeMin"] as Number).toFloat(),
                    (map["pitchRangeMax"] as Number).toFloat()
                ),
                loudnessRange = Pair(
                    (map["loudnessRangeMin"] as Number).toFloat(),
                    (map["loudnessRangeMax"] as Number).toFloat()
                ),
                voicedFramesRatio = (map["voicedFramesRatio"] as Number).toFloat()
            )
        }
    }
}
