#!/bin/bash

# Echo AI PyPI Deployment Script
# This script handles building and deploying the Echo AI package to PyPI

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Globals for temporary project name swap
ORIGINAL_NAME=""
RESTORE_NAME=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to get current project name from pyproject.toml
get_current_name() {
    grep '^name = ' pyproject.toml | cut -d'"' -f2
}

# Function to get current version from pyproject.toml
get_current_version() {
    grep '^version = ' pyproject.toml | cut -d'"' -f2
}

# Function to bump version
bump_version() {
    local version_type=$1
    local current_version=$(get_current_version)
    local major minor patch
    
    IFS='.' read -r major minor patch <<< "$current_version"
    
    case $version_type in
        "patch")
            patch=$((patch + 1))
            ;;
        "minor")
            minor=$((minor + 1))
            patch=0
            ;;
        "major")
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        *)
            print_error "Invalid version type: $version_type"
            exit 1
            ;;
    esac
    
    echo "$major.$minor.$patch"
}

# Function to update project name in pyproject.toml
update_name() {
    local new_name=$1
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^name = \".*\"/name = \"$new_name\"/" pyproject.toml
    else
        # Linux
        sed -i "s/^name = \".*\"/name = \"$new_name\"/" pyproject.toml
    fi
}

# Function to update version in pyproject.toml
update_version() {
    local new_version=$1
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^version = \".*\"/version = \"$new_version\"/" pyproject.toml
    else
        # Linux
        sed -i "s/^version = \".*\"/version = \"$new_version\"/" pyproject.toml
    fi
}

# Ensure original name is restored on exit if we changed it
restore_original_name() {
    if [ "$RESTORE_NAME" = true ] && [ -n "$ORIGINAL_NAME" ]; then
        print_info "Restoring project name to $ORIGINAL_NAME"
        update_name "$ORIGINAL_NAME"
    fi
}

trap restore_original_name EXIT

# Function to create git tag
create_git_tag() {
    local version=$1
    print_info "Creating git tag v$version"
    git add pyproject.toml
    git commit -m "Bump version to $version"
    git tag -a "v$version" -m "Release version $version"
    git push origin main --tags
}

# Function to build and deploy
build_and_deploy() {
    local test_pypi=$1
    
    print_info "Building Echo AI package..."
    
    # Temporarily switch project name to avoid PyPI similarity conflicts
    ORIGINAL_NAME=$(get_current_name)
    local temp_name="echo-app"
    if [ "$ORIGINAL_NAME" != "$temp_name" ]; then
        print_info "Temporarily setting project name to $temp_name for distribution"
        RESTORE_NAME=true
        update_name "$temp_name"
    fi
    
    # Clean previous builds
    rm -rf dist/ build/ *.egg-info/
    
    # Build package
    python -m build
    
    if [ "$test_pypi" = true ]; then
        print_info "Uploading to TestPyPI..."
        python -m twine upload --repository testpypi dist/*
        print_success "Successfully uploaded to TestPyPI!"
        print_info "You can test the package with: pip install --index-url https://test.pypi.org/simple/ echo"
    else
        print_info "Uploading to PyPI..."
        python -m twine upload dist/*
        print_success "Successfully uploaded to PyPI!"
        print_info "You can install the package with: pip install echo-app"
    fi

    # Restore original name explicitly after successful upload
    restore_original_name
    RESTORE_NAME=false
}

# Parse command line arguments
BUMP_VERSION=false
VERSION_TYPE=""
CREATE_TAG=true
TEST_PYPI=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
                    echo
        echo "Echo AI PyPI Deployment Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  -h, --help          Show this help message"
        echo "  -v, --version       Show current version"
        echo "  --patch             Bump patch version (0.1.0 -> 0.1.1)"
        echo "  --minor             Bump minor version (0.1.0 -> 0.2.0)"
        echo "  --major             Bump major version (0.1.0 -> 1.0.0)"
        echo "  --no-bump           Don't bump version"
        echo "  --no-tag            Don't create git tag"
        echo "  --test              Upload to TestPyPI instead of PyPI"
        echo ""
        echo "Examples:"
        echo "  $0 --patch          # Bump patch version and deploy"
        echo "  $0 --minor          # Bump minor version and deploy"
        echo "  $0 --no-bump        # Deploy current version without bumping"
        echo "  $0 --test           # Deploy to TestPyPI"
        echo ""
            exit 0
            ;;
        -v|--version)
            echo "Current version: $(get_current_version)"
            exit 0
            ;;
        --patch|--minor|--major)
            BUMP_VERSION=true
            VERSION_TYPE=${1#--}
            shift
            ;;
        --no-bump)
            BUMP_VERSION=false
            shift
            ;;
        --no-tag)
            CREATE_TAG=false
            shift
            ;;
        --test)
            TEST_PYPI=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main deployment logic
print_info "Starting Echo AI deployment process..."

# If no explicit bump type was provided via flags, interactively ask the user
if [ "$BUMP_VERSION" = false ]; then
    current_version=$(get_current_version)
    # Preview next versions
    next_patch=$(bump_version "patch")
    IFS='.' read -r cv_major cv_minor cv_patch <<< "$current_version"
    next_minor="$cv_major.$((cv_minor + 1)).0"
    next_major="$((cv_major + 1)).0.0"

    echo
    echo "Select version bump type:"
    echo "1) patch ($current_version -> $next_patch)"
    echo "2) minor ($current_version -> $next_minor)"
    echo "3) major ($current_version -> $next_major)"
    echo "4) Skip version bump"
    read -p "Enter choice (1-4): " -n 1 -r
    echo
    case $REPLY in
        1)
            BUMP_VERSION=true
            VERSION_TYPE="patch"
            ;;
        2)
            BUMP_VERSION=true
            VERSION_TYPE="minor"
            ;;
        3)
            BUMP_VERSION=true
            VERSION_TYPE="major"
            ;;
        4)
            BUMP_VERSION=false
            ;;
        *)
            print_error "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository. Please run this script from the project root."
    exit 1
fi

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_warning "There are uncommitted changes in the repository."
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Deployment cancelled."
        exit 0
    fi
fi

# Handle version bumping
if [ "$BUMP_VERSION" = true ]; then
    current_version=$(get_current_version)
    new_version=$(bump_version "$VERSION_TYPE")
    
    print_info "Bumping version from $current_version to $new_version"
    update_version "$new_version"

    # Ask whether to create a git tag unless disabled explicitly via --no-tag
    if [ "$CREATE_TAG" = true ]; then
        read -p "Create git tag v$new_version? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            create_git_tag "$new_version"
        else
            CREATE_TAG=false
            print_info "Skipping git tag creation."
        fi
    fi
else
    new_version=$(get_current_version)
    print_info "Using current version: $new_version"
fi

# Build and deploy
build_and_deploy "$TEST_PYPI"

print_success "Echo AI deployment completed successfully!"
print_info "Version: $new_version"
if [ "$TEST_PYPI" = true ]; then
    print_info "Package: echo (TestPyPI)"
    print_info "TestPyPI URL: https://test.pypi.org/project/echo-app/"
else
    print_info "Package: echo"
    print_info "PyPI URL: https://pypi.org/project/echo-app/"
fi
