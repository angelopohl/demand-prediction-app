"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Upload, Download, BarChart3, TrendingUp, AlertCircle, CheckCircle, Clock } from "lucide-react"
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold text-gray-900 flex items-center justify-center gap-2">
            <TrendingUp className="h-8 w-8 text-blue-600" />
            Predicción de Demanda ML
          </h1>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Analiza y predice la demanda de productos usando algoritmos de Machine Learning optimizados. Máximo 5,000
            filas para mejor rendimiento.
          </p>
        </div>

        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Cargar Datos
            </CardTitle>
            <CardDescription>
              Sube tu archivo CSV (máximo 10MB, 5,000 filas). Columnas requeridas: Warehouse, Product_Category, Year,
              Month, Day, Weekday, Season, Promo, Order_Demand
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="csv-file">Archivo CSV</Label>
              <Input id="csv-file" type="file" accept=".csv" onChange={handleFileChange} disabled={isAnalyzing} />
            </div>

            {file && (
              <Alert>
                <CheckCircle className="h-4 w-4" />
                <AlertDescription>
                  Archivo: {file.name} ({(file.size / 1024).toFixed(1)} KB)
                  {file.size > 5 * 1024 * 1024 && " - Archivo grande, el procesamiento puede tardar más"}
                </AlertDescription>
              </Alert>
            )}

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button onClick={runAnalysis} disabled={!file || isAnalyzing} className="w-full">
              {isAnalyzing ? "Analizando..." : "Ejecutar Análisis"}
            </Button>

            {isAnalyzing && (
              <div className="space-y-2">
                <Progress value={progress} className="w-full" />
                <div className="flex justify-between text-sm text-gray-600">
                  <span>
                    {progress < 25
                      ? "Cargando datos..."
                      : progress < 50
                        ? "Preparando características..."
                        : progress < 75
                          ? "Entrenando modelos..."
                          : "Finalizando..."}
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {timeElapsed}s
                  </span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        {results && (
          <div className="space-y-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
                <BarChart3 className="h-6 w-6" />
                Resultados del Análisis
              </h2>
              <Button onClick={downloadResults} variant="outline">
                <Download className="h-4 w-4 mr-2" />
                Descargar CSV
              </Button>
            </div>

            <ResultsDisplay results={results} />
          </div>
        )}
      </div>
    </div>
  )
}
