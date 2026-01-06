package com.runanywhere.startup_hackathon20

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Voice Feature Repository - JSON-based storage system
 * Handles storage, retrieval, and time-series tracking of voice features
 */
class VoiceFeatureRepository(private val context: Context) {

    companion object {
        private const val TAG = "VoiceFeatureRepo"
        private const val FEATURES_DIR = "voice_features"
        private const val FEATURES_INDEX_FILE = "features_index.json"
        private const val MAX_FEATURES_IN_MEMORY = 1000
    }

    private val featuresDir: File by lazy {
        File(context.filesDir, FEATURES_DIR).apply {
            if (!exists()) mkdirs()
        }
    }

    private val indexFile: File by lazy {
        File(featuresDir, FEATURES_INDEX_FILE)
    }

    /**
     * Save voice features to storage
     */
    fun saveFeatures(features: VoiceFeatures): Boolean {
        return try {
            Log.d(TAG, "Saving features for ${features.recordingId}")

            // Save individual feature file
            val featureFile = File(featuresDir, "${features.recordingId}.json")
            val jsonObject = JSONObject(features.toMap())
            featureFile.writeText(jsonObject.toString(2))

            // Update index
            updateIndex(features)

            Log.d(TAG, "Features saved successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error saving features: ${e.message}", e)
            false
        }
    }

    /**
     * Load features by recording ID
     */
    fun loadFeatures(recordingId: String): VoiceFeatures? {
        return try {
            val featureFile = File(featuresDir, "$recordingId.json")
            if (!featureFile.exists()) {
                Log.w(TAG, "Feature file not found: $recordingId")
                return null
            }

            val jsonString = featureFile.readText()
            val jsonObject = JSONObject(jsonString)
            val map = jsonObject.toMap()

            VoiceFeatures.fromMap(map)
        } catch (e: Exception) {
            Log.e(TAG, "Error loading features: ${e.message}", e)
            null
        }
    }

    /**
     * Load features by file path
     */
    fun loadFeaturesByPath(filePath: String): VoiceFeatures? {
        return try {
            val index = loadIndex()
            val recordingId = index.firstOrNull { it["filePath"] == filePath }?.get("recordingId") as? String
            recordingId?.let { loadFeatures(it) }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading features by path: ${e.message}", e)
            null
        }
    }

    /**
     * Get all features (time-series data)
     */
    fun getAllFeatures(): List<VoiceFeatures> {
        return try {
            val index = loadIndex()
            index.mapNotNull { entry ->
                val recordingId = entry["recordingId"] as? String
                recordingId?.let { loadFeatures(it) }
            }.sortedByDescending { it.timestamp }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading all features: ${e.message}", e)
            emptyList()
        }
    }

    /**
     * Get features within time range
     */
    fun getFeaturesInTimeRange(startTime: Long, endTime: Long): List<VoiceFeatures> {
        return getAllFeatures().filter { it.timestamp in startTime..endTime }
    }

    /**
     * Get latest N features
     */
    fun getLatestFeatures(count: Int): List<VoiceFeatures> {
        return getAllFeatures().take(count)
    }

    /**
     * Delete features by recording ID
     */
    fun deleteFeatures(recordingId: String): Boolean {
        return try {
            val featureFile = File(featuresDir, "$recordingId.json")
            val deleted = featureFile.delete()
            if (deleted) {
                removeFromIndex(recordingId)
            }
            deleted
        } catch (e: Exception) {
            Log.e(TAG, "Error deleting features: ${e.message}", e)
            false
        }
    }

    /**
     * Get feature statistics (for progress monitoring)
     */
    fun getFeatureStatistics(): FeatureStatistics {
        val features = getAllFeatures()
        if (features.isEmpty()) {
            return FeatureStatistics()
        }

        return FeatureStatistics(
            totalRecordings = features.size,
            averageHealthScore = features.map { it.healthScore }.average().toFloat(),
            averagePitch = features.map { it.pitch }.average().toFloat(),
            averageJitter = features.map { it.jitter }.average().toFloat(),
            averageShimmer = features.map { it.shimmer }.average().toFloat(),
            averageHNR = features.map { it.hnr }.average().toFloat(),
            firstRecordingDate = features.last().timestamp,
            lastRecordingDate = features.first().timestamp,
            improvementTrend = calculateImprovementTrend(features)
        )
    }

    /**
     * Export features to CSV
     */
    fun exportToCSV(outputFile: File): Boolean {
        return try {
            val features = getAllFeatures()
            val csv = StringBuilder()

            // Header
            csv.append("RecordingID,Timestamp,Duration,Pitch,Loudness,Jitter,Shimmer,HNR,ZCR,Energy,Entropy,Tempo,")
            csv.append("F1,F2,F3,SpectralCentroid,SpectralRolloff,SpectralFlux,")
            csv.append("Breathiness,Roughness,Tremor,PauseCount,PauseDuration,ArticulationRate,HealthScore\n")

            // Data rows
            for (f in features) {
                csv.append("${f.recordingId},${f.timestamp},${f.duration},${f.pitch},${f.loudness},")
                csv.append("${f.jitter},${f.shimmer},${f.hnr},${f.zcr},${f.energy},${f.entropy},${f.tempo},")
                csv.append("${f.formantF1},${f.formantF2},${f.formantF3},")
                csv.append("${f.spectralCentroid},${f.spectralRolloff},${f.spectralFlux},")
                csv.append("${f.breathiness},${f.roughness},${f.tremor},")
                csv.append("${f.pauseCount},${f.pauseDuration},${f.articulationRate},${f.healthScore}\n")
            }

            outputFile.writeText(csv.toString())
            Log.d(TAG, "Exported ${features.size} features to CSV")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error exporting CSV: ${e.message}", e)
            false
        }
    }

    // ===== Private Helper Functions =====

    private fun updateIndex(features: VoiceFeatures) {
        try {
            val index = loadIndex().toMutableList()

            // Remove existing entry if present
            index.removeAll { (it["recordingId"] as? String) == features.recordingId }

            // Add new entry
            index.add(mapOf(
                "recordingId" to features.recordingId,
                "timestamp" to features.timestamp,
                "filePath" to features.filePath,
                "healthScore" to features.healthScore
            ))

            // Keep only latest MAX_FEATURES_IN_MEMORY entries
            val sorted = index.sortedByDescending { it["timestamp"] as Long }
            val trimmed = sorted.take(MAX_FEATURES_IN_MEMORY)

            // Save index
            val jsonArray = JSONArray(trimmed)
            indexFile.writeText(jsonArray.toString(2))

        } catch (e: Exception) {
            Log.e(TAG, "Error updating index: ${e.message}", e)
        }
    }

    private fun loadIndex(): List<Map<String, Any>> {
        return try {
            if (!indexFile.exists()) {
                return emptyList()
            }

            val jsonString = indexFile.readText()
            val jsonArray = JSONArray(jsonString)

            List(jsonArray.length()) { i ->
                jsonArray.getJSONObject(i).toMap()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading index: ${e.message}", e)
            emptyList()
        }
    }

    private fun removeFromIndex(recordingId: String) {
        try {
            val index = loadIndex().toMutableList()
            index.removeAll { (it["recordingId"] as? String) == recordingId }

            val jsonArray = JSONArray(index)
            indexFile.writeText(jsonArray.toString(2))
        } catch (e: Exception) {
            Log.e(TAG, "Error removing from index: ${e.message}", e)
        }
    }

    private fun calculateImprovementTrend(features: List<VoiceFeatures>): Float {
        if (features.size < 2) return 0f

        val recent = features.take(5).map { it.healthScore }.average()
        val older = features.takeLast(5).map { it.healthScore }.average()

        return (recent - older).toFloat()
    }

    private fun JSONObject.toMap(): Map<String, Any> {
        val map = mutableMapOf<String, Any>()
        keys().forEach { key ->
            val value = get(key)
            map[key] = when (value) {
                is JSONArray -> List(value.length()) { i -> value.get(i) }
                else -> value
            }
        }
        return map
    }

    data class FeatureStatistics(
        val totalRecordings: Int = 0,
        val averageHealthScore: Float = 0f,
        val averagePitch: Float = 0f,
        val averageJitter: Float = 0f,
        val averageShimmer: Float = 0f,
        val averageHNR: Float = 0f,
        val firstRecordingDate: Long = 0L,
        val lastRecordingDate: Long = 0L,
        val improvementTrend: Float = 0f
    )
}
