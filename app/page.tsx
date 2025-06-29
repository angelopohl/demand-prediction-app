"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import {
  Upload,
  Download,
  BarChart3,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Clock,
  Github,
  FileText,
  Play,
  ExternalLink,
  Sparkles,
  Users,
  BookOpen,
} from "lucide-react"
import { ResultsDisplay } from "@/components/results-display"

export default function DemandPredictionApp() {
  const [file, setFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [timeElapsed, setTimeElapsed] = useState(0)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile && selectedFile.type === "text/csv") {
      setFile(selectedFile)
      setError(null)

      // Verificar tamaño del archivo
      if (selectedFile.size > 10 * 1024 * 1024) {
        setError("El archivo es demasiado grande. Máximo 10MB.")
        setFile(null)
        return
      }
    } else {
      setError("Por favor selecciona un archivo CSV válido")
    }
  }

  const runAnalysis = async () => {
    if (!file) {
      setError("Por favor selecciona un archivo CSV")
      return
    }

    setIsAnalyzing(true)
    setProgress(0)
    setError(null)
    setResults(null)
    setTimeElapsed(0)

    // Timer para mostrar tiempo transcurrido
    const startTime = Date.now()
    const timer = setInterval(() => {
      setTimeElapsed(Math.floor((Date.now() - startTime) / 1000))
    }, 1000)

    try {
      // Progreso más rápido
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev < 85) return prev + 25
          return prev
        })
      }, 300)

      const formData = new FormData()
      formData.append("file", file)

      console.log("Enviando archivo para análisis...")

      // Timeout más corto
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 30000) // 30 segundos

      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      })

      clearTimeout(timeoutId)
      clearInterval(progressInterval)
      clearInterval(timer)
      setProgress(100)

      if (!response.ok) {
        const errorText = await response.text()
        console.error("Error response:", errorText)

        try {
          const errorData = JSON.parse(errorText)
          throw new Error(errorData.error || "Error en el análisis")
        } catch (parseError) {
          throw new Error(`Error del servidor (${response.status})`)
        }
      }

      const analysisResults = await response.json()

      if (!analysisResults.success) {
        throw new Error(analysisResults.error || "Error en el procesamiento")
      }

      setResults(analysisResults)
    } catch (err) {
      clearInterval(timer)
      console.error("Error en análisis:", err)

      if (err instanceof Error && err.name === "AbortError") {
        setError("El análisis tardó demasiado tiempo. Intenta con un archivo más pequeño.")
      } else {
        setError(err instanceof Error ? err.message : "Error durante el análisis")
      }
    } finally {
      setIsAnalyzing(false)
    }
  }

  const downloadResults = () => {
    if (!results) return

    const csvData = results.categoryAnalysis
      .map((cat: any) => `${cat.category},${cat.avgDemand.toFixed(2)},${cat.error.toFixed(2)},${cat.count}`)
      .join("\n")

    const csvContent = `data:text/csv;charset=utf-8,Category,Avg_Demand,Avg_Error,Count\n${csvData}`
    const encodedUri = encodeURI(csvContent)
    const link = document.createElement("a")
    link.setAttribute("href", encodedUri)
    link.setAttribute("download", "analisis_demanda_resultados.csv")
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // Funciones para los nuevos botones
  const openGitHubRepo = () => {
    window.open("https://github.com/angelopohl/demand-prediction-app", "_blank")
  }

  const downloadDocumentation = () => {
    try {
      // URL directa al archivo PDF en GitHub (raw)
      const githubPdfUrl = "https://github.com/angelopohl/demand-prediction-app/raw/master/OPCION%20PROYECTO%20AE.pdf"

      // Crear enlace para descargar desde GitHub
      const link = document.createElement("a")
      link.href = githubPdfUrl
      link.download = "OPCION_PROYECTO_AE.pdf"
      link.target = "_blank" // Abrir en nueva pestaña como respaldo
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    } catch (error) {
      console.error("Error descargando documentación:", error)
      // Fallback: abrir directamente en GitHub
      window.open(
        "https://github.com/angelopohl/demand-prediction-app/blob/master/OPCION%20PROYECTO%20AE.pdf",
        "_blank",
      )
    }
  }

  const openYouTubeTutorial = () => {
    // Por ahora muestra un mensaje, después se cambiará por la URL real
    alert("Tutorial de YouTube próximamente disponible")
    // window.open("URL_DEL_VIDEO_YOUTUBE", "_blank")
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header mejorado con logo de la universidad */}
      <div className="bg-white/95 backdrop-blur-sm shadow-lg border-b border-blue-100">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex flex-col lg:flex-row justify-between items-center gap-6">
            {/* Logo y título de la universidad */}
            <div className="flex flex-col lg:flex-row items-center gap-4">
              <div className="flex items-center gap-3">
                {/* Logo UPAO placeholder */}
                <div className="w-16 h-16 bg-gradient-to-br from-blue-600 to-blue-800 rounded-full flex items-center justify-center shadow-lg">
                  <img
                    src="/placeholder.svg?height=64&width=64"
                    alt="UPAO Logo"
                    className="w-12 h-12 rounded-full bg-white p-1"
                  />
                </div>
                <div className="text-center lg:text-left">
                  <h1 className="text-2xl lg:text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                    Predicción de Demanda ML
                  </h1>
                  <p className="text-sm text-gray-600 font-medium">
                    Universidad Privada Antenor Orrego - Trujillo, Perú
                  </p>
                </div>
              </div>
            </div>

            {/* Botones de navegación mejorados */}
            <div className="flex flex-wrap justify-center gap-3">
              <div className="group relative">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={openGitHubRepo}
                  className="flex items-center gap-2 hover:bg-gradient-to-r hover:from-gray-50 hover:to-blue-50 transition-all duration-300 border-blue-200 hover:border-blue-300 hover:shadow-md bg-transparent"
                >
                  <Github className="h-4 w-4 text-gray-700" />
                  <span className="hidden sm:inline font-medium">Repositorio GitHub</span>
                  <span className="sm:hidden font-medium">GitHub</span>
                  <ExternalLink className="h-3 w-3 text-blue-500" />
                </Button>
                <div className="absolute -bottom-12 left-1/2 transform -translate-x-1/2 bg-gray-800 text-white text-xs px-3 py-1 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap z-10">
                  <Users className="inline h-3 w-3 mr-1" />
                  ¡Aporta con tu trabajo en nuestro proyecto!
                </div>
              </div>

              <div className="group relative">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={downloadDocumentation}
                  className="flex items-center gap-2 hover:bg-gradient-to-r hover:from-green-50 hover:to-emerald-50 transition-all duration-300 border-green-200 hover:border-green-300 hover:shadow-md bg-transparent"
                >
                  <FileText className="h-4 w-4 text-gray-700" />
                  <span className="hidden sm:inline font-medium">Documentación</span>
                  <span className="sm:hidden font-medium">Docs</span>
                  <Download className="h-3 w-3 text-green-500" />
                </Button>
                <div className="absolute -bottom-12 left-1/2 transform -translate-x-1/2 bg-gray-800 text-white text-xs px-3 py-1 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap z-10">
                  <BookOpen className="inline h-3 w-3 mr-1" />
                  Descarga la guía completa del proyecto
                </div>
              </div>

              <div className="group relative">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={openYouTubeTutorial}
                  className="flex items-center gap-2 hover:bg-gradient-to-r hover:from-red-50 hover:to-pink-50 transition-all duration-300 border-red-200 hover:border-red-300 hover:shadow-md bg-transparent"
                >
                  <Play className="h-4 w-4 text-gray-700" />
                  <span className="hidden sm:inline font-medium">Tutorial YouTube</span>
                  <span className="sm:hidden font-medium">Tutorial</span>
                  <ExternalLink className="h-3 w-3 text-red-500" />
                </Button>
                <div className="absolute -bottom-12 left-1/2 transform -translate-x-1/2 bg-gray-800 text-white text-xs px-3 py-1 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 whitespace-nowrap z-10">
                  <Sparkles className="inline h-3 w-3 mr-1" />
                  Aprende paso a paso con nuestro tutorial
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Contenido principal mejorado */}
      <div className="max-w-7xl mx-auto p-6 space-y-8">
        {/* Hero Section mejorada */}
        <div className="text-center space-y-4 pt-8 pb-4">
          <div className="inline-flex items-center gap-2 bg-blue-100 text-blue-800 px-4 py-2 rounded-full text-sm font-medium mb-4">
            <TrendingUp className="h-4 w-4" />
            Inteligencia Artificial Aplicada
          </div>
          <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 leading-tight">
            Análisis Predictivo de
            <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent block">
              Demanda con Machine Learning
            </span>
          </h2>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Utiliza algoritmos avanzados de Machine Learning para predecir la demanda de productos con precisión.
            Optimizado para análisis de hasta 5,000 registros con rendimiento superior.
          </p>
        </div>

        {/* Upload Section mejorada */}
        <Card className="shadow-xl border-0 bg-white/80 backdrop-blur-sm">
          <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-t-lg">
            <CardTitle className="flex items-center gap-3 text-xl">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Upload className="h-6 w-6 text-blue-600" />
              </div>
              Cargar Datos para Análisis
            </CardTitle>
            <CardDescription className="text-base">
              Sube tu archivo CSV (máximo 10MB, 5,000 filas). El sistema requiere las siguientes columnas:{" "}
              <span className="font-mono text-sm bg-gray-100 px-2 py-1 rounded">
                Warehouse, Product_Category, Year, Month, Day, Weekday, Season, Promo, Order_Demand
              </span>
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6 p-6">
            <div className="space-y-3">
              <Label htmlFor="csv-file" className="text-base font-medium">
                Seleccionar Archivo CSV
              </Label>
              <div className="relative">
                <Input
                  id="csv-file"
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  disabled={isAnalyzing}
                  className="h-12 text-base border-2 border-dashed border-gray-300 hover:border-blue-400 transition-colors"
                />
              </div>
            </div>

            {file && (
              <Alert className="border-green-200 bg-green-50">
                <CheckCircle className="h-5 w-5 text-green-600" />
                <AlertDescription className="text-base">
                  <strong>Archivo cargado:</strong> {file.name} ({(file.size / 1024).toFixed(1)} KB)
                  {file.size > 5 * 1024 * 1024 && (
                    <span className="block text-amber-600 mt-1">
                      ⚠️ Archivo grande detectado - El procesamiento puede tardar más tiempo
                    </span>
                  )}
                </AlertDescription>
              </Alert>
            )}

            {error && (
              <Alert variant="destructive" className="border-red-200">
                <AlertCircle className="h-5 w-5" />
                <AlertDescription className="text-base">{error}</AlertDescription>
              </Alert>
            )}

            <Button
              onClick={runAnalysis}
              disabled={!file || isAnalyzing}
              className="w-full h-12 text-base font-medium bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-xl"
            >
              {isAnalyzing ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Analizando Datos...
                </>
              ) : (
                <>
                  <BarChart3 className="h-5 w-5 mr-2" />
                  Ejecutar Análisis Predictivo
                </>
              )}
            </Button>

            {isAnalyzing && (
              <div className="space-y-4 p-4 bg-blue-50 rounded-lg">
                <Progress value={progress} className="w-full h-3" />
                <div className="flex justify-between items-center text-sm">
                  <span className="flex items-center gap-2 font-medium text-blue-700">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                    {progress < 25
                      ? "Cargando y validando datos..."
                      : progress < 50
                        ? "Preparando características..."
                        : progress < 75
                          ? "Entrenando modelos ML..."
                          : "Generando resultados..."}
                  </span>
                  <span className="flex items-center gap-1 text-gray-600">
                    <Clock className="h-4 w-4" />
                    {timeElapsed}s transcurridos
                  </span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Section mejorada */}
        {results && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 p-6 bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-200">
              <div>
                <h2 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                  <div className="p-2 bg-green-100 rounded-lg">
                    <BarChart3 className="h-7 w-7 text-green-600" />
                  </div>
                  Resultados del Análisis
                </h2>
                <p className="text-green-700 mt-1">Análisis completado exitosamente</p>
              </div>
              <Button
                onClick={downloadResults}
                variant="outline"
                className="bg-white hover:bg-green-50 border-green-300 hover:border-green-400 transition-all duration-300"
              >
                <Download className="h-4 w-4 mr-2" />
                Exportar Resultados CSV
              </Button>
            </div>

            <ResultsDisplay results={results} />
          </div>
        )}
      </div>
    </div>
  )
}
