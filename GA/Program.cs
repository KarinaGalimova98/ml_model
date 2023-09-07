using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Threading;
using System.Reflection;

public class Nanoparticle
{
    public double Feature1 { get; set; }
    public double Feature2 { get; set; }
    public double Fitness { get; set; }
}

public class GeneticAlgorithm
{
    private readonly Random random = new Random();

    public List<Nanoparticle> InitializePopulation(int populationSize)
    {
        var population = new List<Nanoparticle>();

        for (int i = 0; i < populationSize; i++)
        {
            var nanoparticle = new Nanoparticle
            {
                Feature1 = random.NextDouble(), // Замените на генерацию признаков по вашим требованиям
                Feature2 = random.NextDouble()
            };
            population.Add(nanoparticle);
        }

        return population;
    }

    private double CalculateFitness(Nanoparticle nanoparticle, MLContext mlContext, ITransformer model)
    {
        // Use the trained machine learning model to make predictions for the nanoparticle
        var predictionEngine = mlContext.Model.CreatePredictionEngine<NanoparticleData, NanoparticlePrediction>(model);

        var nanoparticleData = new NanoparticleData

        {
            Feature1 = (float)nanoparticle.Feature1,
            Feature2 = (float)nanoparticle.Feature2
            // Add other features if necessary
        };

        // Make a prediction using the machine learning model
        var prediction = predictionEngine.Predict(nanoparticleData);

        // Calculate the fitness based on the prediction (lower values indicate better cytotoxicity)
        double fitness = -prediction.Cytotoxicity;

        return fitness;
    }

    public List<Nanoparticle> EvolvePopulation(List<Nanoparticle> population, int eliteCount)
    {
        var sortedPopulation = population.OrderByDescending(n => n.Fitness).ToList();

        var newPopulation = new List<Nanoparticle>();

        for (int i = 0; i < eliteCount; i++)
        {
            newPopulation.Add(sortedPopulation[i]); // Элитные наночастицы сохраняются
        }

        while (newPopulation.Count < population.Count)
        {
            var parent1 = SelectParent(population);
            var parent2 = SelectParent(population);
            var child = Crossover(parent1, parent2);
            Mutate(child);
            newPopulation.Add(child);
        }

        return newPopulation;
    }

    private Nanoparticle SelectParent(List<Nanoparticle> population)
    {
        // Простой метод выбора родителя, можно заменить на другие стратегии
        int index = random.Next(population.Count);
        return population[index];
    }

    private Nanoparticle Crossover(Nanoparticle parent1, Nanoparticle parent2)
    {
        // Определите операцию кроссовера между двумя родителями
        // Например, можно взять случайное значение от каждого родителя
        var child = new Nanoparticle
        {
            Feature1 = random.NextDouble() < 0.5 ? parent1.Feature1 : parent2.Feature1,
            Feature2 = random.NextDouble() < 0.5 ? parent1.Feature2 : parent2.Feature2
        };
        return child;
    }

    private void Mutate(Nanoparticle nanoparticle)
    {
        // Реализуйте операцию мутации для наночастицы
        double mutationRate = 0.1;

        // Мутация может изменять признаки наночастицы случайным образом
        // Это помогает разнообразить популяцию и исследовать новые решения
        if (random.NextDouble() < mutationRate)
        {
            nanoparticle.Feature1 = random.NextDouble();
        }
        if (random.NextDouble() < mutationRate)
        {
            nanoparticle.Feature2 = random.NextDouble();
        }
    }

    public void RunGeneticAlgorithm(int populationSize, int generations)
    {
        var population = InitializePopulation(populationSize);
        int eliteCount = 2; // Количество элитных наночастиц, которые будут сохранены на каждой итерации
        double mutationRate = 0.1; // Вероятность мутации

        for (int generation = 0; generation < generations; generation++)
        {
            foreach (var nanoparticle in population)
            {
                nanoparticle.Fitness = CalculateFitness(nanoparticle, mlContext, model);
            }

            population = EvolvePopulation(population, eliteCount);

            var bestNanoparticle = population.OrderByDescending(n => n.Fitness).First();
            Console.WriteLine($"Generation {generation}: Best Fitness = {bestNanoparticle.Fitness}");
        }
    }
}

class Program
{
    static void Main(string[] args)
    {
        var geneticAlgorithm = new GeneticAlgorithm();
        geneticAlgorithm.RunGeneticAlgorithm(populationSize: 100, generations: 50);
    }
}