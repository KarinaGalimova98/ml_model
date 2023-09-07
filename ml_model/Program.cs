using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

// Определение класса для хранения данных
public class NanoparticleData
{
    [LoadColumn(0)] public float Feature1;
    [LoadColumn(1)] public float Feature2;
    // Добавьте другие признаки, если они есть в вашем наборе данных
    [LoadColumn(2)] public float Cytotoxicity;
}

// Определение класса для прогноза
public class NanoparticlePrediction
{
    [ColumnName("Score")] public float Cytotoxicity;
}

class Program
{
    static void Main(string[] args)
    {
        // Создание объекта MLContext для работы с ML.NET
        var context = new MLContext();

        // Загрузка данных из CSV файла
        var data = context.Data.LoadFromTextFile<NanoparticleData>("path/to/your/dataset.csv", separatorChar: ',');

        // Определение столбца с целевой переменной и признаков
        var pipeline = context.Transforms.CopyColumns("Label", "Cytotoxicity")
            .Append(context.Transforms.Concatenate("Features", "Feature1", "Feature2")) // Добавьте другие признаки
            
            .Append(context.Transforms.NormalizeMinMax("Label"));

        // Обучение модели
        var model = pipeline.Fit(data);

        // Тестирование модели
        var testSample = new NanoparticleData
        {
            Feature1 = 0.5f,
            Feature2 = 0.3f
            // Установите значения других признаков для тестирования
        };

        var predictor = context.Model.CreatePredictionEngine<NanoparticleData, NanoparticlePrediction>(model);
        var prediction = predictor.Predict(testSample);

        Console.WriteLine($"Predicted Cytotoxicity: {prediction.Cytotoxicity}");
    }
}
