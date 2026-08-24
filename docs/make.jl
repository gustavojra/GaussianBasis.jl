using Documenter
using GaussianBasis

DocMeta.setdocmeta!(GaussianBasis, :DocTestSetup, :(using GaussianBasis); recursive=true)

makedocs(;
    modules=[GaussianBasis],
    authors="gustavojra <gustavo.aroeira@gmail.com> and contributors",
    repo="https://github.com/FermiQC/GaussianBasis.jl/blob/{commit}{path}#{line}",
    sitename="GaussianBasis.jl",
    checkdocs=:exports,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://fermiqc.github.io/GaussianBasis.jl",
        edit_link="main",
        assets=String[],
        sidebar_sitename=false,
    ),
    pages=[
        "Home" => "index.md",
        "One-Electron Integrals" => "oneelectron.md",
        "Two-Electron Integrals" => "eri.md",
        "Gradients" => "gradients.md",
        "Hessians" => "hessians.md",
    ],
)

deploydocs(;
    repo="github.com/FermiQC/GaussianBasis.jl",
    devbranch="main",
)
