package com.runanywhere.startup_hackathon20

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverter
import androidx.room.TypeConverters
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

// --- Data class for the feature set ---
// This is separate from the entity to keep the core logic clean.
data class VoiceAnalysisResult(
    val audioFilePath: String,
    val timestamp: Long = System.currentTimeMillis(),
    val pitch: Float = 0f, val loudness: Float = 0f, val jitter: Float = 0f, val shimmer: Float = 0f,
    val hnr: Float = 0f, val zcr: Float = 0f, val formants: List<Float> = emptyList(),
    val energy: Float = 0f, val entropy: Float = 0f, val breathiness: Float = 0f, val roughness: Float = 0f,
    val tremor: Float = 0f, val articulationRate: Float = 0f, val tempo: Float = 0f,
    val pauseCount: Int = 0, val averagePauseDuration: Float = 0f, val mfcc: List<Float> = emptyList(),
    val chromagram: List<Float> = emptyList(), val spectralCentroid: Float = 0f,
    val spectralRolloff: Float = 0f, val spectralFlux: Float = 0f, val healthScore: Int = 0
)

// --- Database Entity ---
@Entity(tableName = "voice_analysis_results")
@TypeConverters(VoiceAnalysisResultConverters::class)
data class VoiceAnalysisResultEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val audioFilePath: String, val timestamp: Long, val pitch: Float, val loudness: Float,
    val jitter: Float, val shimmer: Float, val hnr: Float, val zcr: Float, val formants: List<Float>,
    val energy: Float, val entropy: Float, val breathiness: Float, val roughness: Float,
    val tremor: Float, val articulationRate: Float, val tempo: Float, val pauseCount: Int,
    val averagePauseDuration: Float, val mfcc: List<Float>, val chromagram: List<Float>,
    val spectralCentroid: Float, val spectralRolloff: Float, val spectralFlux: Float, val healthScore: Int
) {
    // Converter from the main data class to the database entity
    constructor(result: VoiceAnalysisResult) : this(
        audioFilePath = result.audioFilePath, timestamp = result.timestamp, pitch = result.pitch,
        loudness = result.loudness, jitter = result.jitter, shimmer = result.shimmer, hnr = result.hnr,
        zcr = result.zcr, formants = result.formants, energy = result.energy, entropy = result.entropy,
        breathiness = result.breathiness, roughness = result.roughness, tremor = result.tremor,
        articulationRate = result.articulationRate, tempo = result.tempo, pauseCount = result.pauseCount,
        averagePauseDuration = result.averagePauseDuration, mfcc = result.mfcc,
        chromagram = result.chromagram, spectralCentroid = result.spectralCentroid,
        spectralRolloff = result.spectralRolloff, spectralFlux = result.spectralFlux, healthScore = result.healthScore
    )
}

// --- Data Access Object (DAO) ---
@Dao
interface VoiceAnalysisResultDao {
    @Insert
    suspend fun insert(result: VoiceAnalysisResultEntity)

    @Query("SELECT * FROM voice_analysis_results ORDER BY timestamp DESC")
    suspend fun getAllResults(): List<VoiceAnalysisResultEntity>
}

// --- Type Converters for Room ---
class VoiceAnalysisResultConverters {
    private val json = Json

    @TypeConverter
    fun fromFloatList(list: List<Float>): String {
        return json.encodeToString(list)
    }

    @TypeConverter
    fun toFloatList(jsonString: String): List<Float> {
        return json.decodeFromString(jsonString)
    }
}

// --- Database Class ---
@Database(entities = [VoiceAnalysisResultEntity::class], version = 1, exportSchema = false)
@TypeConverters(VoiceAnalysisResultConverters::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun voiceAnalysisResultDao(): VoiceAnalysisResultDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "speech_twin_database"
                ).fallbackToDestructiveMigration().build()
                INSTANCE = instance
                instance
            }
        }
    }
}
