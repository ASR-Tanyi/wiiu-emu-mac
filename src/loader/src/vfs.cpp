#include <wiiu/common/log.hpp>
#include <wiiu/loader/vfs.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <fstream>

namespace wiiu::loader {
namespace {

using wiiu::log::Channel;

} // namespace

std::string normalize_path(std::string_view path) {
    std::vector<std::string_view> parts;

    std::size_t start = 0;
    while (start <= path.size()) {
        const std::size_t end = path.find('/', start);
        const std::string_view part = path.substr(
            start, end == std::string_view::npos ? std::string_view::npos : end - start);

        if (part == "..") {
            // Dropping instead of escaping: the caller is guest code, and a
            // path that climbs out of its mount is not a path we should serve.
            if (!parts.empty()) {
                parts.pop_back();
            }
        } else if (!part.empty() && part != ".") {
            parts.push_back(part);
        }

        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }

    std::string out = "/";
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) {
            out += '/';
        }
        out += parts[i];
    }
    return out;
}

// --------------------------------------------------------------------------
// HostMount
// --------------------------------------------------------------------------

std::optional<std::filesystem::path> HostMount::resolve(std::string_view path) const {
    const std::string normalized = normalize_path(path);

    std::filesystem::path candidate = root_;
    if (normalized != "/") {
        candidate /= normalized.substr(1);
    }

    // Belt and braces: normalize_path already refuses to climb out, but the
    // host path is checked too, in case a symlink inside the directory points
    // somewhere else.
    std::error_code ec;
    const auto real_root = std::filesystem::weakly_canonical(root_, ec);
    const auto real_candidate = std::filesystem::weakly_canonical(candidate, ec);
    if (ec) {
        return candidate; // nothing there yet; the path itself is still safe
    }

    const auto root_string = real_root.string();
    const auto candidate_string = real_candidate.string();
    if (!candidate_string.starts_with(root_string)) {
        WIIU_LOG_WARN(Loader, "refusing path '{}': it escapes the mount", path);
        return std::nullopt;
    }
    return real_candidate;
}

bool HostMount::exists(std::string_view path) const {
    const auto resolved = resolve(path);
    if (!resolved) {
        return false;
    }
    std::error_code ec;
    return std::filesystem::exists(*resolved, ec);
}

bool HostMount::is_directory(std::string_view path) const {
    const auto resolved = resolve(path);
    if (!resolved) {
        return false;
    }
    std::error_code ec;
    return std::filesystem::is_directory(*resolved, ec);
}

std::optional<u64> HostMount::size(std::string_view path) const {
    const auto resolved = resolve(path);
    if (!resolved) {
        return std::nullopt;
    }
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(*resolved, ec);
    if (ec) {
        return std::nullopt;
    }
    return bytes;
}

bool HostMount::read(std::string_view path, std::vector<u8>& out) const {
    const auto resolved = resolve(path);
    if (!resolved) {
        return false;
    }

    std::ifstream file(*resolved, std::ios::binary | std::ios::ate);
    if (!file) {
        return false;
    }
    const auto bytes = static_cast<std::streamsize>(file.tellg());
    file.seekg(0);

    out.assign(static_cast<std::size_t>(bytes), 0);
    return static_cast<bool>(file.read(reinterpret_cast<char*>(out.data()), bytes));
}

std::vector<FileInfo> HostMount::list(std::string_view path) const {
    std::vector<FileInfo> out;
    const auto resolved = resolve(path);
    if (!resolved) {
        return out;
    }

    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(*resolved, ec)) {
        FileInfo info;
        info.name = entry.path().filename().string();
        info.is_directory = entry.is_directory(ec);
        info.size = info.is_directory ? 0 : entry.file_size(ec);
        out.push_back(std::move(info));
    }

    // Deterministic order: directory iteration is not, and a game that walks
    // its content directory should see the same thing every run.
    std::ranges::sort(out,
                      [](const FileInfo& lhs, const FileInfo& rhs) { return lhs.name < rhs.name; });
    return out;
}

// --------------------------------------------------------------------------
// VirtualFileSystem
// --------------------------------------------------------------------------

void VirtualFileSystem::mount(std::string prefix, std::shared_ptr<Mount> source) {
    std::string normalized = normalize_path(prefix);
    unmount(normalized);
    mounts_.push_back({std::move(normalized), std::move(source)});

    // Longest first, so find() can stop at the first match.
    std::ranges::sort(mounts_, [](const Entry& lhs, const Entry& rhs) {
        return lhs.prefix.size() > rhs.prefix.size();
    });
}

void VirtualFileSystem::unmount(std::string_view prefix) {
    const std::string normalized = normalize_path(prefix);
    std::erase_if(mounts_,
                  [&normalized](const Entry& entry) { return entry.prefix == normalized; });
}

bool VirtualFileSystem::mount_title_directory(const std::filesystem::path& directory,
                                              std::string& error) {
    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec)) {
        error = fmt::format("{} is not a directory", directory.string());
        return false;
    }

    // The three directories a decrypted title always has. Their absence means
    // this is some other folder, and mounting it would only fail later in a
    // more confusing place.
    for (const char* required : {"code", "content", "meta"}) {
        if (!std::filesystem::is_directory(directory / required, ec)) {
            error = fmt::format("{} has no {}/ directory", directory.string(), required);
            return false;
        }
    }

    mount("/vol/code", std::make_shared<HostMount>(directory / "code"));
    mount("/vol/content", std::make_shared<HostMount>(directory / "content"));
    mount("/vol/meta", std::make_shared<HostMount>(directory / "meta"));

    WIIU_LOG_INFO(Loader, "mounted title directory {}", directory.string());
    return true;
}

const VirtualFileSystem::Entry* VirtualFileSystem::find(std::string_view path,
                                                        std::string& relative) const {
    const std::string normalized = normalize_path(path);

    for (const Entry& entry : mounts_) {
        if (!normalized.starts_with(entry.prefix)) {
            continue;
        }
        // "/vol/contentious" must not match a "/vol/content" mount.
        if (normalized.size() > entry.prefix.size() && normalized[entry.prefix.size()] != '/') {
            continue;
        }
        relative = normalized.substr(entry.prefix.size());
        if (relative.empty()) {
            relative = "/";
        }
        return &entry;
    }
    return nullptr;
}

bool VirtualFileSystem::exists(std::string_view path) const {
    std::string relative;
    const Entry* const entry = find(path, relative);
    return entry != nullptr && entry->source->exists(relative);
}

bool VirtualFileSystem::is_directory(std::string_view path) const {
    std::string relative;
    const Entry* const entry = find(path, relative);
    return entry != nullptr && entry->source->is_directory(relative);
}

std::optional<u64> VirtualFileSystem::size(std::string_view path) const {
    std::string relative;
    const Entry* const entry = find(path, relative);
    return entry == nullptr ? std::nullopt : entry->source->size(relative);
}

bool VirtualFileSystem::read(std::string_view path, std::vector<u8>& out) const {
    std::string relative;
    const Entry* const entry = find(path, relative);
    return entry != nullptr && entry->source->read(relative, out);
}

std::vector<FileInfo> VirtualFileSystem::list(std::string_view path) const {
    std::string relative;
    const Entry* const entry = find(path, relative);
    return entry == nullptr ? std::vector<FileInfo>{} : entry->source->list(relative);
}

std::vector<std::string> VirtualFileSystem::mount_points() const {
    std::vector<std::string> out;
    out.reserve(mounts_.size());
    for (const Entry& entry : mounts_) {
        out.push_back(entry.prefix);
    }
    std::ranges::sort(out);
    return out;
}

} // namespace wiiu::loader
